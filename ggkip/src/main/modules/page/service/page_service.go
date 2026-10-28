package service

import (
	"bytes"
	"context"

	"ggkip/src/main/modules/page/model"
	"ggkip/src/main/modules/page/repo"
)

// Change is the dedup decision for a fetched page.
type Change int

const (
	ChangeUnchanged Change = iota
	ChangeNew
	ChangeChanged
)

// Status is the page status stored for this change.
func (c Change) Status() string {
	switch c {
	case ChangeNew:
		return model.StatusNew
	case ChangeChanged:
		return model.StatusChanged
	default:
		return model.StatusUnchanged
	}
}

// Classify compares the stored hash with the hash of the freshly extracted
// text. A page that never got a record (nil hash) counts as new.
func Classify(stored, current []byte) Change {
	switch {
	case stored == nil:
		return ChangeNew
	case bytes.Equal(stored, current):
		return ChangeUnchanged
	default:
		return ChangeChanged
	}
}

type Service struct {
	repo *repo.Repo
}

func New(r *repo.Repo) *Service {
	return &Service{repo: r}
}

func (s *Service) List(ctx context.Context, f model.ListFilter) ([]model.Page, error) {
	return s.repo.List(ctx, f)
}
