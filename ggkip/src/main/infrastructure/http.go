package infrastructure

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"
)

var (
	// ErrNotFound is returned by repositories when a row does not exist.
	ErrNotFound = errors.New("not found")
	// ErrInvalidInput marks errors caused by bad client input.
	ErrInvalidInput = errors.New("invalid input")
)

// SHA256 hashes a string; used for url_hash and content_hash.
func SHA256(s string) []byte {
	sum := sha256.Sum256([]byte(s))
	return sum[:]
}

// ListResponse is the envelope for every list endpoint. NextCursor is passed
// back as ?cursor= to get the next page; it is null on the last page.
type ListResponse[T any] struct {
	Items      []T    `json:"items"`
	NextCursor *int64 `json:"next_cursor"`
}

// NewListResponse trims the extra row a repository fetched (limit+1) and sets
// the cursor from the last returned item.
func NewListResponse[T any](items []T, limit int, id func(T) int64) ListResponse[T] {
	resp := ListResponse[T]{Items: items}
	if len(items) > limit {
		resp.Items = items[:limit]
		next := id(items[limit-1])
		resp.NextCursor = &next
	}
	if resp.Items == nil {
		resp.Items = []T{}
	}
	return resp
}

// Page is cursor pagination input. Cursor is the id to continue after (0 = start).
type Page struct {
	Cursor int64
	Limit  int
}

func ParsePage(r *http.Request, defaultLimit, maxLimit int) (Page, error) {
	p := Page{Limit: defaultLimit}
	cursor, err := QueryInt64(r, "cursor")
	if err != nil {
		return p, err
	}
	if cursor != nil {
		p.Cursor = *cursor
	}
	if v := r.URL.Query().Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			return p, fmt.Errorf("%w: limit must be a positive integer", ErrInvalidInput)
		}
		p.Limit = min(n, maxLimit)
	}
	return p, nil
}

func QueryInt64(r *http.Request, key string) (*int64, error) {
	v := r.URL.Query().Get(key)
	if v == "" {
		return nil, nil
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("%w: %s must be an integer", ErrInvalidInput, key)
	}
	return &n, nil
}

func QueryString(r *http.Request, key string) *string {
	v := strings.TrimSpace(r.URL.Query().Get(key))
	if v == "" {
		return nil
	}
	return &v
}

func QueryBool(r *http.Request, key string) (*bool, error) {
	v := r.URL.Query().Get(key)
	if v == "" {
		return nil, nil
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return nil, fmt.Errorf("%w: %s must be true or false", ErrInvalidInput, key)
	}
	return &b, nil
}

// QueryTime accepts RFC 3339 timestamps or plain YYYY-MM-DD dates.
func QueryTime(r *http.Request, key string) (*time.Time, error) {
	v := r.URL.Query().Get(key)
	if v == "" {
		return nil, nil
	}
	for _, layout := range []string{time.RFC3339, time.DateOnly} {
		if t, err := time.Parse(layout, v); err == nil {
			return &t, nil
		}
	}
	return nil, fmt.Errorf("%w: %s must be RFC 3339 or YYYY-MM-DD", ErrInvalidInput, key)
}

func PathID(r *http.Request) (int64, error) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	if err != nil || id <= 0 {
		return 0, fmt.Errorf("%w: id must be a positive integer", ErrInvalidInput)
	}
	return id, nil
}

func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// WriteError maps ErrNotFound to 404 and ErrInvalidInput to 400; anything
// else is logged and returned as 500 without internal details.
func WriteError(w http.ResponseWriter, log *slog.Logger, err error) {
	switch {
	case errors.Is(err, ErrNotFound):
		WriteJSON(w, http.StatusNotFound, map[string]string{"error": err.Error()})
	case errors.Is(err, ErrInvalidInput):
		WriteJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
	default:
		log.Error("request failed", "error", err)
		WriteJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal error"})
	}
}
