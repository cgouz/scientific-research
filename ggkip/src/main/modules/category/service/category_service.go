package service

import (
	"context"

	"ggkip/src/main/modules/category/model"
	"ggkip/src/main/modules/category/repo"
)

type Service struct {
	repo *repo.Repo
}

func New(r *repo.Repo) *Service {
	return &Service{repo: r}
}

func (s *Service) List(ctx context.Context, f model.ListFilter) ([]model.Category, error) {
	return s.repo.List(ctx, f)
}

func (s *Service) Get(ctx context.Context, id int64) (model.Category, error) {
	return s.repo.Get(ctx, id)
}

// Active returns the active taxonomy, used to build the LLM classification prompt.
func (s *Service) Active(ctx context.Context) ([]model.Category, error) {
	active := true
	return s.repo.List(ctx, model.ListFilter{IsActive: &active})
}
