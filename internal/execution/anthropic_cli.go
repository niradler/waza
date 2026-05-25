package execution

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
type AnthropicCLIEngine struct {
	defaultModelID string
	binPath        string
	workspace      string
	keepWorkspace  bool
	mtx            sync.Mutex
	initCalled     atomic.Bool
}

// NewAnthropicCLIEngine returns an engine that invokes `claude` for each Execute.
// If claudeBin is empty, the engine looks up `claude` on PATH at Initialize time.
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

// claudeJSONResult mirrors the `--output-format json` schema from the Claude Code CLI.
type claudeJSONResult struct {
	Type          string `json:"type"`
	Subtype       string `json:"subtype"`
	IsError       bool   `json:"is_error"`
	DurationMs    int64  `json:"duration_ms"`
	DurationAPIMs int64  `json:"duration_api_ms"`
	NumTurns      int    `json:"num_turns"`
	Result        string `json:"result"`
	SessionID     string `json:"session_id"`
	TotalCostUSD  float64 `json:"total_cost_usd"`
	Usage         struct {
		InputTokens             int `json:"input_tokens"`
		CacheCreationInputToks  int `json:"cache_creation_input_tokens"`
		CacheReadInputToks      int `json:"cache_read_input_tokens"`
		OutputTokens            int `json:"output_tokens"`
	} `json:"usage"`
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

	// Workspace handling — reuse if provided, otherwise create + seed resources.
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

	// Build system prompt: skill body (if requested) + instructions.
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
		"--output-format", "json",
		"--permission-mode", "bypassPermissions",
		"--no-session-persistence",
		"--add-dir", workspaceDir,
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

	out, runErr := cmd.Output()
	durMs := time.Since(start).Milliseconds()

	resp := &ExecutionResponse{
		Events:       []copilot.SessionEvent{},
		ModelID:      modelID,
		DurationMs:   durMs,
		ToolCalls:    []models.ToolCall{},
		WorkspaceDir: workspaceDir,
	}
	if !req.SkipWorkspaceCapture {
		resp.WorkspaceFiles = captureWorkspaceFiles(workspaceDir)
	}

	// claude --print --output-format json emits the result envelope on stdout
	// even when it reports an error (is_error: true) and exits non-zero. Try
	// to parse stdout first; fall back to surfacing the exec error only when
	// stdout doesn't contain a JSON envelope.
	var parsed claudeJSONResult
	parseErr := json.Unmarshal(out, &parsed)

	if parseErr == nil {
		resp.FinalOutput = parsed.Result
		resp.SessionID = parsed.SessionID
		resp.Success = !parsed.IsError
		if parsed.IsError {
			resp.ErrorMsg = parsed.Result
		}
	} else if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			resp.ErrorMsg = fmt.Sprintf("claude exited %d: %s", exitErr.ExitCode(), strings.TrimSpace(string(exitErr.Stderr)))
		} else {
			resp.ErrorMsg = runErr.Error()
		}
		resp.Success = false
		return resp, nil
	} else {
		resp.ErrorMsg = fmt.Sprintf("parse claude json: %v; raw=%s", parseErr, truncate(string(out), 512))
		resp.Success = false
		return resp, nil
	}
	resp.Usage = &models.UsageStats{
		Turns:            parsed.NumTurns,
		InputTokens:      parsed.Usage.InputTokens,
		OutputTokens:     parsed.Usage.OutputTokens,
		CacheReadTokens:  parsed.Usage.CacheReadInputToks,
		CacheWriteTokens: parsed.Usage.CacheCreationInputToks,
	}

	return resp, nil
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

// skillDirs replicates the relevant subset of CopilotEngine.getSkillDirs.
// It returns the directories to scan for SKILL.md files when building the
// system prompt for a task.
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

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "...(truncated)"
}
