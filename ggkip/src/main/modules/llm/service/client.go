package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"ggkip/src/main/modules/llm/dto"
)

// Supported runtimes (--llm-runtime / LLM_RUNTIME).
const (
	RuntimeVLLM     = "vllm"
	RuntimeLlamaCpp = "llamacpp"
)

// Client turns clean page text into the strict output JSON. Implementations
// only talk to the model server; validation happens in the Normalizer.
type Client interface {
	Normalize(ctx context.Context, in dto.NormalizeInput, schema map[string]any) (dto.NormalizeOutput, error)
	Runtime() string
	Model() string
}

type ClientConfig struct {
	BaseURL   string
	Model     string
	APIKey    string
	Timeout   time.Duration
	MaxTokens int
}

// NewClient selects the implementation for runtime.
func NewClient(runtime string, cfg ClientConfig) (Client, error) {
	switch runtime {
	case RuntimeVLLM:
		return NewVLLMClient(cfg), nil
	case RuntimeLlamaCpp:
		return NewLlamaCppClient(cfg), nil
	default:
		return nil, fmt.Errorf("unknown llm runtime %q (use %s or %s)", runtime, RuntimeVLLM, RuntimeLlamaCpp)
	}
}

// RetryableError marks failures worth retrying: timeouts, connection errors,
// 429 and 5xx responses, and replies that were not valid JSON.
type RetryableError struct{ Err error }

// ErrTruncated means the reply hit max_tokens. Repeating the same request
// gives the same reply, so the Normalizer retries without the content field.
var ErrTruncated = errors.New("llm output was cut off at max_tokens")

func (e *RetryableError) Error() string { return e.Err.Error() }
func (e *RetryableError) Unwrap() error { return e.Err }

// chatClient is the OpenAI-compatible /chat/completions call that both vLLM
// and llama-server expose. The runtimes differ only in request extras.
type chatClient struct {
	cfg  ClientConfig
	http *http.Client
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func newChatClient(cfg ClientConfig) chatClient {
	return chatClient{cfg: cfg, http: &http.Client{Timeout: cfg.Timeout}}
}

// complete sends body (which must include messages) and returns the
// assistant message text.
func (c chatClient) complete(ctx context.Context, body map[string]any) (string, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	endpoint := strings.TrimRight(c.cfg.BaseURL, "/") + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", &RetryableError{Err: fmt.Errorf("llm request: %w", err)}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return "", &RetryableError{Err: fmt.Errorf("llm response: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		err := fmt.Errorf("llm status %d: %s", resp.StatusCode, truncate(string(raw), 500))
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			return "", &RetryableError{Err: err}
		}
		return "", err
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("llm response is not chat completion JSON: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", &RetryableError{Err: errors.New("llm returned no choices")}
	}
	if out.Choices[0].FinishReason == "length" {
		return "", &RetryableError{Err: ErrTruncated}
	}
	return out.Choices[0].Message.Content, nil
}

func (c chatClient) messages(in dto.NormalizeInput) ([]chatMessage, error) {
	user, err := json.Marshal(in)
	if err != nil {
		return nil, err
	}
	return []chatMessage{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: string(user)},
	}, nil
}

func (c chatClient) normalize(ctx context.Context, in dto.NormalizeInput, body map[string]any) (dto.NormalizeOutput, error) {
	msgs, err := c.messages(in)
	if err != nil {
		return dto.NormalizeOutput{}, err
	}
	body["model"] = c.cfg.Model
	body["messages"] = msgs
	body["temperature"] = 0
	body["max_tokens"] = c.cfg.MaxTokens
	// Qwen3-style templates otherwise emit a <think> block before the JSON.
	body["chat_template_kwargs"] = map[string]any{"enable_thinking": false}

	text, err := c.complete(ctx, body)
	if err != nil {
		return dto.NormalizeOutput{}, err
	}
	out, err := ParseOutput(text)
	if err != nil {
		return out, &RetryableError{Err: err}
	}
	return out, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
