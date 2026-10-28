package service

import (
	"context"

	"ggkip/src/main/modules/llm/dto"
)

// LlamaCppClient talks to llama-server's OpenAI-compatible endpoint. Output
// is constrained with response_format json_object + schema, which llama.cpp
// turns into a grammar.
type LlamaCppClient struct {
	chat chatClient
}

func NewLlamaCppClient(cfg ClientConfig) *LlamaCppClient {
	return &LlamaCppClient{chat: newChatClient(cfg)}
}

func (c *LlamaCppClient) Runtime() string { return RuntimeLlamaCpp }
func (c *LlamaCppClient) Model() string   { return c.chat.cfg.Model }

func (c *LlamaCppClient) Normalize(ctx context.Context, in dto.NormalizeInput, schema map[string]any) (dto.NormalizeOutput, error) {
	return c.chat.normalize(ctx, in, map[string]any{
		"response_format": map[string]any{
			"type":   "json_object",
			"schema": schema,
		},
		"cache_prompt": true,
	})
}
