package handler

import (
	"errors"
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawl/dto"
	"ggkip/src/main/modules/crawl/model"
	"ggkip/src/main/modules/crawl/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/sites/{id}/crawl", h.trigger)
	mux.HandleFunc("GET /v1/crawl-runs", h.list)
	mux.HandleFunc("GET /v1/crawl-runs/{id}", h.get)
}

// trigger starts a crawl in the background and returns 202 with the new run;
// follow it with GET /v1/crawl-runs/{id}.
func (h *Handler) trigger(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	run, err := h.svc.TriggerSite(r.Context(), id)
	switch {
	case errors.Is(err, service.ErrAlreadyRunning):
		infrastructure.WriteJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
	case errors.Is(err, service.ErrBusy):
		infrastructure.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"error": err.Error()})
	case err != nil:
		infrastructure.WriteError(w, h.log, err)
	default:
		infrastructure.WriteJSON(w, http.StatusAccepted, dto.FromModel(run))
	}
}

// list supports ?site_id=, ?status=, ?cursor=, ?limit= (newest first).
func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	page, err := infrastructure.ParsePage(r, 50, 500)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	f := model.ListFilter{Status: infrastructure.QueryString(r, "status"), Cursor: page.Cursor, Limit: page.Limit}
	if f.SiteID, err = infrastructure.QueryInt64(r, "site_id"); err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	items, err := h.svc.List(r.Context(), f)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	out := make([]dto.CrawlRunResponse, len(items))
	for i, run := range items {
		out[i] = dto.FromModel(run)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, page.Limit, func(r dto.CrawlRunResponse) int64 { return r.ID }))
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	run, err := h.svc.Get(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	infrastructure.WriteJSON(w, http.StatusOK, dto.FromModel(run))
}
