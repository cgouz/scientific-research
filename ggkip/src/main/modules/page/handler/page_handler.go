package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/page/dto"
	"ggkip/src/main/modules/page/model"
	"ggkip/src/main/modules/page/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/pages", h.list)
}

// list supports ?site_id=, ?status=, ?cursor=, ?limit= (newest first).
func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	page, err := infrastructure.ParsePage(r, 100, 1000)
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
	out := make([]dto.PageResponse, len(items))
	for i, p := range items {
		out[i] = dto.FromModel(p)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, page.Limit, func(p dto.PageResponse) int64 { return p.ID }))
}
