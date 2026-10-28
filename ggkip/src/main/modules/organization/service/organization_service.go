package service

import (
	"context"

	"ggkip/src/main/modules/organization/model"
	"ggkip/src/main/modules/organization/repo"
)

type Service struct {
	repo *repo.Repo
}

func New(r *repo.Repo) *Service {
	return &Service{repo: r}
}

func (s *Service) List(ctx context.Context, f model.ListFilter) ([]model.Organization, error) {
	return s.repo.List(ctx, f)
}

func (s *Service) Get(ctx context.Context, id int64) (model.Organization, error) {
	return s.repo.Get(ctx, id)
}
