package graders

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/microsoft/waza/internal/models"
	"github.com/santhosh-tekuri/jsonschema/v6"
)

// jsonSchemaGrader validates that the agent output is valid JSON matching a given schema.
type jsonSchemaGrader struct {
	name       string
	schema     map[string]any
	schemaFile string
}

// NewJSONSchemaGrader creates a [jsonSchemaGrader] that validates agent output against
// a JSON schema provided inline or via a file path.
func NewJSONSchemaGrader(name string, args models.JSONSchemaGraderParameters) (*jsonSchemaGrader, error) {
	if args.Schema == nil && args.SchemaFile == "" {
		return nil, fmt.Errorf("json_schema grader '%s' must have either 'schema' or 'schema_file'", name)
	}

	return &jsonSchemaGrader{
		name:       name,
		schema:     args.Schema,
		schemaFile: args.SchemaFile,
	}, nil
}

func (jsg *jsonSchemaGrader) Name() string            { return jsg.name }
func (jsg *jsonSchemaGrader) Kind() models.GraderKind { return models.GraderKindJSONSchema }

func (jsg *jsonSchemaGrader) Grade(ctx context.Context, gradingContext *Context) (*models.GraderResults, error) {
	return measureTime(func() (*models.GraderResults, error) {
		// Step 1: check if the output is valid JSON. Falls back to a fenced
		// ```json``` block when the candidate wraps JSON in a markdown report.
		var outputValue any
		candidate := extractJSONPayload(gradingContext.Output)
		if err := json.Unmarshal([]byte(candidate), &outputValue); err != nil {
			return &models.GraderResults{
				Name:     jsg.name,
				Type:     models.GraderKindJSONSchema,
				Score:    0.0,
				Passed:   false,
				Feedback: fmt.Sprintf("Output is not valid JSON: %v", err),
				Details: map[string]any{
					"error": err.Error(),
				},
			}, nil
		}

		// Step 2: resolve the schema
		schemaMap, err := jsg.resolveSchema()
		if err != nil {
			return nil, fmt.Errorf("json_schema grader '%s': %w", jsg.name, err)
		}

		// Step 3: validate against schema
		failures, err := validateAgainstSchema(outputValue, schemaMap)
		if err != nil {
			return nil, fmt.Errorf("json_schema grader '%s': %w", jsg.name, err)
		}

		if len(failures) > 0 {
			return &models.GraderResults{
				Name:     jsg.name,
				Type:     models.GraderKindJSONSchema,
				Score:    0.0,
				Passed:   false,
				Feedback: strings.Join(failures, "; "),
				Details: map[string]any{
					"failures": failures,
				},
			}, nil
		}

		return &models.GraderResults{
			Name:     jsg.name,
			Type:     models.GraderKindJSONSchema,
			Score:    1.0,
			Passed:   true,
			Feedback: "Output matches JSON schema",
		}, nil
	})
}

// resolveSchema returns the schema map, loading from file if necessary.
func (jsg *jsonSchemaGrader) resolveSchema() (map[string]any, error) {
	if jsg.schema != nil {
		return jsg.schema, nil
	}

	data, err := os.ReadFile(jsg.schemaFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read schema file %q: %w", jsg.schemaFile, err)
	}

	var schemaMap map[string]any
	if err := json.Unmarshal(data, &schemaMap); err != nil {
		return nil, fmt.Errorf("failed to parse schema file %q: %w", jsg.schemaFile, err)
	}

	return schemaMap, nil
}

// validateAgainstSchema validates the given value against a JSON schema map.
func validateAgainstSchema(value any, schemaMap map[string]any) ([]string, error) {
	schemaJSON, err := json.Marshal(schemaMap)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize schema: %w", err)
	}

	var schemaValue any
	if err := json.Unmarshal(schemaJSON, &schemaValue); err != nil {
		return nil, fmt.Errorf("failed to parse schema for validation: %w", err)
	}

	compiler := jsonschema.NewCompiler()
	if err := compiler.AddResource("schema.json", schemaValue); err != nil {
		return nil, fmt.Errorf("failed to add schema resource: %w", err)
	}

	schema, err := compiler.Compile("schema.json")
	if err != nil {
		return nil, fmt.Errorf("failed to compile JSON schema: %w", err)
	}

	if err := schema.Validate(value); err != nil {
		var failures []string
		failures = append(failures, fmt.Sprintf("Schema validation failed: %v", err))
		return failures, nil
	}

	return nil, nil
}

// extractJSONPayload returns a JSON-parseable substring from output. It first
// tries the raw output; if that fails, it scans for a fenced ```json``` (or
// bare ```) block; if still nothing, it falls back to the longest balanced
// {...} or [...] segment in the text.
func extractJSONPayload(out string) string {
	trimmed := strings.TrimSpace(out)
	if trimmed == "" {
		return trimmed
	}
	var probe any
	if json.Unmarshal([]byte(trimmed), &probe) == nil {
		return trimmed
	}
	if block := findFencedJSON(trimmed); block != "" {
		return block
	}
	if block := findBalancedJSON(trimmed); block != "" {
		return block
	}
	return trimmed
}

func findFencedJSON(s string) string {
	for _, marker := range []string{"```json", "```JSON", "```"} {
		idx := strings.Index(s, marker)
		if idx < 0 {
			continue
		}
		rest := s[idx+len(marker):]
		end := strings.Index(rest, "```")
		if end < 0 {
			continue
		}
		candidate := strings.TrimSpace(rest[:end])
		if candidate == "" {
			continue
		}
		var probe any
		if json.Unmarshal([]byte(candidate), &probe) == nil {
			return candidate
		}
	}
	return ""
}

func findBalancedJSON(s string) string {
	for _, open := range []byte{'{', '['} {
		close := byte('}')
		if open == '[' {
			close = ']'
		}
		start := strings.IndexByte(s, open)
		for start >= 0 {
			depth := 0
			inStr := false
			esc := false
			for i := start; i < len(s); i++ {
				c := s[i]
				if inStr {
					if esc {
						esc = false
						continue
					}
					if c == '\\' {
						esc = true
						continue
					}
					if c == '"' {
						inStr = false
					}
					continue
				}
				switch c {
				case '"':
					inStr = true
				case open:
					depth++
				case close:
					depth--
					if depth == 0 {
						candidate := s[start : i+1]
						var probe any
						if json.Unmarshal([]byte(candidate), &probe) == nil {
							return candidate
						}
					}
				}
			}
			next := strings.IndexByte(s[start+1:], open)
			if next < 0 {
				break
			}
			start = start + 1 + next
		}
	}
	return ""
}
