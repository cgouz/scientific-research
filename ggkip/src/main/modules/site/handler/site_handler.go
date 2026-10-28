package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/site/dto"
	"ggkip/src/main/modules/site/model"
	"ggkip/src/main/modules/site/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

// Register adds the read endpoints; POST /v1/sites/{id}/crawl lives in the crawl module.
func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/sites", h.list)
	mux.HandleFunc("GET /v1/sites/{id}", h.get)
}

// list supports ?country_id=, ?organization_id=, ?status=, ?cursor=, ?limit=.
func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	page, err := infrastructure.ParsePage(r, 100, 1000)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	f := model.ListFilter{Status: infrastructure.QueryString(r, "status"), Cursor: page.Cursor, Limit: page.Limit}
	if f.CountryID, err = infrastructure.QueryInt64(r, "country_id"); err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	if f.OrganizationID, err = infrastructure.QueryInt64(r, "organization_id"); err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	items, err := h.svc.List(r.Context(), f)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	out := make([]dto.SiteResponse, len(items))
	for i, s := range items {
		out[i] = dto.FromModel(s)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, page.Limit, func(s dto.SiteResponse) int64 { return s.ID }))
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	s, err := h.svc.Get(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	infrastructure.WriteJSON(w, http.StatusOK, dto.FromModel(s))
}
