package service

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"
	"unicode/utf8"

	"ggkip/src/main/modules/llm/dto"
	"ggkip/src/main/modules/llm/model"
)

// Normalizer wraps a Client with input truncation, a global concurrency
// limit, validation and retries. It is shared by all crawl runs.
type Normalizer struct {
	client        Client
	sem           chan struct{}
	maxRetries    int
	maxInputChars int
	retryDelay    time.Duration // multiplied by the attempt number
	log           *slog.Logger
}

func NewNormalizer(client Client, workers, maxRetries, maxInputChars int, log *slog.Logger) *Normalizer {
	return &Normalizer{
		client:        client,
		sem:           make(chan struct{}, workers),
		maxRetries:    maxRetries,
		maxInputChars: maxInputChars,
		retryDelay:    time.Second,
		log:           log,
	}
}

func (n *Normalizer) Runtime() string { return n.client.Runtime() }
func (n *Normalizer) Model() string   { return n.client.Model() }

// Normalize calls the LLM and validates the reply. Transport errors, invalid
// JSON and taxonomy violations are retried up to maxRetries times.
func (n *Normalizer) Normalize(ctx context.Context, in dto.NormalizeInput, t *model.Taxonomy) (model.Result, error) {
	in.Content = truncateRunes(in.Content, n.maxInputChars)
	in.Categories = t.Categories
	in.RecordTypes = t.RecordTypes
	schema := OutputSchema(t)

	select {
	case n.sem <- struct{}{}:
	case <-ctx.Done():
		return model.Result{}, ctx.Err()
	}
	defer func() { <-n.sem }()

	var lastErr error
	for attempt := 0; attempt <= n.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return model.Result{}, ctx.Err()
			case <-time.After(time.Duration(attempt) * n.retryDelay):
			}
		}
		start := time.Now()
		out, err := n.client.Normalize(ctx, in, schema)
		if err != nil {
			lastErr = err
			n.log.Warn("llm call failed", "url", in.SourceURL, "attempt", attempt+1,
				"llm_runtime", n.client.Runtime(), "duration", time.Since(start), "error", err)
			var retryable *RetryableError
			if !errors.As(err, &retryable) {
				return model.Result{}, err
			}
			continue
		}
		res, err := Validate(out, t)
		if err != nil {
			lastErr = err
			n.log.Warn("llm output rejected", "url", in.SourceURL, "attempt", attempt+1, "error", err)
			continue
		}
		n.log.Debug("llm normalized page", "url", in.SourceURL, "llm_runtime", n.client.Runtime(),
			"duration", time.Since(start), "category", out.CategoryCode, "record_type", res.RecordType)
		return res, nil
	}
	return model.Result{}, fmt.Errorf("llm failed after %d attempts: %w", n.maxRetries+1, lastErr)
}

func truncateRunes(s string, max int) string {
	if utf8.RuneCountInString(s) <= max {
		return s
	}
	r := []rune(s)
	return string(r[:max])
}
