package execution

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	copilot "github.com/github/copilot-sdk/go"
	"github.com/microsoft/waza/internal/models"
)

// AnthropicCLIEngine executes evaluation tasks by shelling out to the
// Claude Code CLI (`claude`). It reuses Claude Code's agent loop (tool use,
// multi-turn, MCP) instead of reimplementing it in Go. Auth via ANTHROPIC_API_KEY.
//
// The executor speaks `--output-format stream-json` so it can harvest tool_use
// and tool_result events into resp.ToolCalls/resp.SkillInvocations — that's
// what powers the tool_calls / tool_constraint / skill_invocation / behavior /
// action_sequence graders. The final `{"type":"result"}` line carries the
// same envelope the legacy `--output-format json` path used.
type AnthropicCLIEngine struct {
	defaultModelID string
	binPath        string
	workspace      string
	keepWorkspace  bool
	mtx            sync.Mutex
	initCalled     atomic.Bool
}

func NewAnthropicCLIEngine(modelID, claudeBin string) *AnthropicCLIEngine {
	return &AnthropicCLIEngine{defaultModelID: modelID, binPath: claudeBin}
}

func (e *AnthropicCLIEngine) SetKeepWorkspace(keep bool) { e.keepWorkspace = keep }

func (e *AnthropicCLIEngine) Initialize(ctx context.Context) error {
	if e.binPath == "" {
		path, err := exec.LookPath("claude")
		if err != nil {
			return fmt.Errorf("anthropic-cli: `claude` not found on PATH: %w", err)
		}
		e.binPath = path
	}
	e.initCalled.Store(true)
	return nil
}

// claudeStreamLine is the union envelope for `--output-format stream-json`.
// Only the fields we consume are modeled.
type claudeStreamLine struct {
	Type      string          `json:"type"`
	Subtype   string          `json:"subtype,omitempty"`
	SessionID string          `json:"session_id,omitempty"`
	Message   json.RawMessage `json:"message,omitempty"`
	// result envelope
	IsError      bool    `json:"is_error,omitempty"`
	DurationMs   int64   `json:"duration_ms,omitempty"`
	NumTurns     int     `json:"num_turns,omitempty"`
	Result       string  `json:"result,omitempty"`
	TotalCostUSD float64 `json:"total_cost_usd,omitempty"`
	Usage        struct {
		InputTokens            int `json:"input_tokens"`
		CacheCreationInputToks int `json:"cache_creation_input_tokens"`
		CacheReadInputToks     int `json:"cache_read_input_tokens"`
		OutputTokens           int `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

type claudeAssistantMessage struct {
	Content []claudeMessageBlock `json:"content"`
}

type claudeMessageBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type claudeUserMessage struct {
	Content []claudeMessageBlock `json:"content"`
}

func (e *AnthropicCLIEngine) Execute(ctx context.Context, req *ExecutionRequest) (*ExecutionResponse, error) {
	if !e.initCalled.Load() {
		return nil, fmt.Errorf("AnthropicCLIEngine.Execute: Initialize must be called first")
	}
	if req == nil {
		return nil, fmt.Errorf("AnthropicCLIEngine.Execute: nil request")
	}

	e.mtx.Lock()
	defer e.mtx.Unlock()

	start := time.Now()

	var workspaceDir string
	if req.WorkspaceDir != "" {
		workspaceDir = req.WorkspaceDir
	} else {
		if e.workspace != "" {
			_ = os.RemoveAll(e.workspace)
			e.workspace = ""
		}
		tmpDir, err := os.MkdirTemp("", "waza-anthropic-*")
		if err != nil {
			return nil, fmt.Errorf("create workspace: %w", err)
		}
		e.workspace = tmpDir
		workspaceDir = tmpDir
		if err := setupWorkspaceResources(workspaceDir, req.Resources); err != nil {
			return nil, fmt.Errorf("seed workspace: %w", err)
		}
	}

	var systemParts []string
	if !req.NoSkills {
		skillDirs := e.skillDirs(req)
		if msg := buildSkillSystemMessage(skillDirs, req.SkillName, !req.SuppressSkillBody); msg != "" {
			systemParts = append(systemParts, msg)
		}
	}
	if msg := buildInstructionSystemMessage(req.Instructions); msg != "" {
		systemParts = append(systemParts, msg)
	}
	systemPrompt := strings.Join(systemParts, "\n")

	modelID := req.ModelID
	if modelID == "" {
		modelID = e.defaultModelID
	}
	modelID = normalizeAnthropicModelID(modelID)

	timeout := req.Timeout
	if timeout == 0 {
		timeout = 5 * time.Minute
	}
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	args := []string{
		"--print",
		"--bare",
		"--output-format", "stream-json",
		"--verbose",
		"--permission-mode", "bypassPermissions",
		"--add-dir", workspaceDir,
	}
	// Ephemeral graders should never persist. Task runs allow persistence so
	// callers can chain multi-turn sessions via SessionID.
	if req.EphemeralSession {
		args = append(args, "--no-session-persistence")
	}
	if modelID != "" {
		args = append(args, "--model", modelID)
	}
	if systemPrompt != "" {
		args = append(args, "--system-prompt", systemPrompt)
	}
	if req.SessionID != "" {
		args = append(args, "--resume", req.SessionID)
	}

	cmd := exec.CommandContext(execCtx, e.binPath, args...)
	cmd.Stdin = strings.NewReader(req.Message)
	cmd.Dir = workspaceDir

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start claude: %w", err)
	}

	resp := &ExecutionResponse{
		Events:           []copilot.SessionEvent{},
		ModelID:          modelID,
		ToolCalls:        []models.ToolCall{},
		SkillInvocations: []SkillInvocation{},
		WorkspaceDir:     workspaceDir,
	}

	finalEnvelope, parseErr := streamParse(stdout, resp)
	waitErr := cmd.Wait()

	resp.DurationMs = time.Since(start).Milliseconds()
	if !req.SkipWorkspaceCapture {
		resp.WorkspaceFiles = captureWorkspaceFiles(workspaceDir)
	}

	if finalEnvelope != nil {
		resp.FinalOutput = finalEnvelope.Result
		resp.SessionID = finalEnvelope.SessionID
		resp.Success = !finalEnvelope.IsError
		if finalEnvelope.IsError {
			resp.ErrorMsg = finalEnvelope.Result
		}
		resp.Usage = &models.UsageStats{
			Turns:            finalEnvelope.NumTurns,
			InputTokens:      finalEnvelope.Usage.InputTokens,
			OutputTokens:     finalEnvelope.Usage.OutputTokens,
			CacheReadTokens:  finalEnvelope.Usage.CacheReadInputToks,
			CacheWriteTokens: finalEnvelope.Usage.CacheCreationInputToks,
		}
	} else {
		// No result line — surface whatever we got from stderr / exit error.
		resp.Success = false
		if waitErr != nil {
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				resp.ErrorMsg = fmt.Sprintf("claude exited %d: %s", exitErr.ExitCode(), strings.TrimSpace(stderrBuf.String()))
			} else {
				resp.ErrorMsg = waitErr.Error()
			}
		} else if parseErr != nil {
			resp.ErrorMsg = fmt.Sprintf("parse claude stream: %v", parseErr)
		} else if stderrBuf.Len() > 0 {
			resp.ErrorMsg = strings.TrimSpace(stderrBuf.String())
		} else {
			resp.ErrorMsg = "claude produced no result envelope"
		}
	}

	return resp, nil
}

// streamParse reads claude's stream-json output and populates tool/skill
// invocations on resp. The final `result` line is returned so the caller can
// fill in totals (session id, usage, success flag).
func streamParse(r io.Reader, resp *ExecutionResponse) (*claudeStreamLine, error) {
	pending := map[string]*models.ToolCall{}
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 1<<20), 64<<20)

	var finalLine *claudeStreamLine

	for scanner.Scan() {
		raw := scanner.Bytes()
		if len(bytes.TrimSpace(raw)) == 0 {
			continue
		}
		var line claudeStreamLine
		if err := json.Unmarshal(raw, &line); err != nil {
			continue
		}

		switch line.Type {
		case "assistant":
			if len(line.Message) == 0 {
				continue
			}
			var msg claudeAssistantMessage
			if err := json.Unmarshal(line.Message, &msg); err != nil {
				continue
			}
			for _, block := range msg.Content {
				if block.Type != "tool_use" || block.ID == "" {
					continue
				}
				args := decodeToolArgs(block.Input)
				tc := &models.ToolCall{
					Name:      block.Name,
					Arguments: args,
				}
				pending[block.ID] = tc
				if block.Name == "Skill" && args.Skill != "" {
					resp.SkillInvocations = append(resp.SkillInvocations, SkillInvocation{Name: args.Skill})
				}
			}
		case "user":
			if len(line.Message) == 0 {
				continue
			}
			var msg claudeUserMessage
			if err := json.Unmarshal(line.Message, &msg); err != nil {
				continue
			}
			for _, block := range msg.Content {
				if block.Type != "tool_result" || block.ToolUseID == "" {
					continue
				}
				tc, ok := pending[block.ToolUseID]
				if !ok {
					continue
				}
				tc.Success = !block.IsError
				resp.ToolCalls = append(resp.ToolCalls, *tc)
				delete(pending, block.ToolUseID)
			}
		case "result":
			cp := line
			finalLine = &cp
		}
	}
	// Any tool_use that never got a tool_result still counts as an attempted call.
	for _, tc := range pending {
		tc.Success = false
		resp.ToolCalls = append(resp.ToolCalls, *tc)
	}

	if err := scanner.Err(); err != nil {
		return finalLine, fmt.Errorf("scan stream: %w", err)
	}
	return finalLine, nil
}

func decodeToolArgs(input json.RawMessage) models.ToolCallArgs {
	if len(input) == 0 {
		return models.ToolCallArgs{}
	}
	var raw struct {
		Path        string `json:"path"`
		FilePath    string `json:"file_path"`
		FileText    string `json:"file_text"`
		Content     string `json:"content"`
		Command     string `json:"command"`
		Description string `json:"description"`
		Skill       string `json:"skill"`
		SkillName   string `json:"skill_name"`
	}
	_ = json.Unmarshal(input, &raw)
	path := raw.Path
	if path == "" {
		path = raw.FilePath
	}
	fileText := raw.FileText
	if fileText == "" {
		fileText = raw.Content
	}
	skill := raw.Skill
	if skill == "" {
		skill = raw.SkillName
	}
	return models.ToolCallArgs{
		Path:        path,
		FileText:    fileText,
		Command:     raw.Command,
		Description: raw.Description,
		Skill:       skill,
	}
}

func (e *AnthropicCLIEngine) Shutdown(ctx context.Context) error {
	if e.workspace == "" {
		return nil
	}
	if e.keepWorkspace {
		fmt.Fprintf(os.Stderr, "Workspace preserved: %s\n", e.workspace)
	} else if err := os.RemoveAll(e.workspace); err != nil {
		return fmt.Errorf("remove workspace %s: %w", e.workspace, err)
	}
	e.workspace = ""
	return nil
}

func (e *AnthropicCLIEngine) SessionUsage(sessionID string) *models.UsageStats { return nil }

func (e *AnthropicCLIEngine) skillDirs(req *ExecutionRequest) []string {
	var dirs []string
	if req.SourceDir != "" {
		dirs = append(dirs, req.SourceDir)
	}
	dirs = append(dirs, req.SkillPaths...)
	return dirs
}

// normalizeAnthropicModelID maps friendly waza model aliases to the dated
// model IDs the Anthropic API expects. Unknown IDs are passed through.
func normalizeAnthropicModelID(id string) string {
	switch id {
	case "claude-haiku-4.5", "claude-haiku-4-5":
		return "claude-haiku-4-5"
	case "claude-sonnet-4.6", "claude-sonnet-4-6":
		return "claude-sonnet-4-6"
	case "claude-opus-4.7", "claude-opus-4-7":
		return "claude-opus-4-7"
	default:
		return id
	}
}
