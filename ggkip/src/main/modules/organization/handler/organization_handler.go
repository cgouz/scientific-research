package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/organization/dto"
	"ggkip/src/main/modules/organization/model"
	"ggkip/src/main/modules/organization/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/organizations", h.list)
	mux.HandleFunc("GET /v1/organizations/{id}", h.get)
}

// list supports ?country_id=, ?organization_type_id=, ?parent_id=, ?status=, ?cursor=, ?limit=.
func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	f, err := parseFilter(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	items, err := h.svc.List(r.Context(), f)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	out := make([]dto.OrganizationResponse, len(items))
	for i, o := range items {
		out[i] = dto.FromModel(o)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, f.Limit, func(o dto.OrganizationResponse) int64 { return o.ID }))
}

func parseFilter(r *http.Request) (model.ListFilter, error) {
	page, err := infrastructure.ParsePage(r, 100, 1000)
	if err != nil {
		return model.ListFilter{}, err
	}
	f := model.ListFilter{Status: infrastructure.QueryString(r, "status"), Cursor: page.Cursor, Limit: page.Limit}
	for key, dst := range map[string]**int64{
		"country_id": &f.CountryID, "organization_type_id": &f.OrganizationTypeID, "parent_id": &f.ParentID,
	} {
		if *dst, err = infrastructure.QueryInt64(r, key); err != nil {
			return f, err
		}
	}
	return f, nil
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	o, err := h.svc.Get(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	infrastructure.WriteJSON(w, http.StatusOK, dto.FromModel(o))
}
