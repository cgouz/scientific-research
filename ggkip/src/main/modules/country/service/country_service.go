package service

import (
	"context"

	"ggkip/src/main/modules/country/model"
	"ggkip/src/main/modules/country/repo"
)

type Service struct {
	repo *repo.Repo
}

func New(r *repo.Repo) *Service {
	return &Service{repo: r}
}

func (s *Service) List(ctx context.Context, f model.ListFilter) ([]model.Country, error) {
	return s.repo.List(ctx, f)
}

func (s *Service) Get(ctx context.Context, id int64) (model.Country, error) {
	return s.repo.Get(ctx, id)
}

func (s *Service) Translations(ctx context.Context, id int64) ([]model.Translation, error) {
	return s.repo.Translations(ctx, id)
}
