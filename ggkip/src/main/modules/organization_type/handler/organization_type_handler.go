package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/organization_type/dto"
	"ggkip/src/main/modules/organization_type/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/organization-types", h.list)
}

func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	items, err := h.svc.List(r.Context())
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	out := make([]dto.OrganizationTypeResponse, len(items))
	for i, t := range items {
		out[i] = dto.FromModel(t)
	}
	infrastructure.WriteJSON(w, http.StatusOK, infrastructure.ListResponse[dto.OrganizationTypeResponse]{Items: out})
}
