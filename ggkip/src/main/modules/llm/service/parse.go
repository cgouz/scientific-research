package service

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"ggkip/src/main/modules/llm/dto"
)

// ParseOutput extracts the JSON object from a model reply. It tolerates a
// <think> block and ```json fences, which some models add despite instructions.
func ParseOutput(raw string) (dto.NormalizeOutput, error) {
	var out dto.NormalizeOutput
	s := strings.TrimSpace(raw)
	if i := strings.LastIndex(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	}
	start, end := strings.IndexByte(s, '{'), strings.LastIndexByte(s, '}')
	if start < 0 || end <= start {
		return out, errors.New("llm reply contains no JSON object")
	}
	if err := json.Unmarshal([]byte(s[start:end+1]), &out); err != nil {
		return out, fmt.Errorf("llm reply is not valid JSON: %w", err)
	}
	return out, nil
}
