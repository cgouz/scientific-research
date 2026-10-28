package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/country/dto"
	"ggkip/src/main/modules/country/model"
	"ggkip/src/main/modules/country/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/countries", h.list)
	mux.HandleFunc("GET /v1/countries/{id}", h.get)
}

// list supports ?region=, ?is_active=, ?locale=uz|ru|kaa, ?cursor=, ?limit=.
func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	page, err := infrastructure.ParsePage(r, 300, 1000)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	active, err := infrastructure.QueryBool(r, "is_active")
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	items, err := h.svc.List(r.Context(), model.ListFilter{
		Region:   infrastructure.QueryString(r, "region"),
		IsActive: active,
		Locale:   infrastructure.QueryString(r, "locale"),
		Cursor:   page.Cursor,
		Limit:    page.Limit,
	})
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	out := make([]dto.CountryResponse, len(items))
	for i, c := range items {
		out[i] = dto.FromModel(c)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, page.Limit, func(c dto.CountryResponse) int64 { return c.ID }))
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	c, err := h.svc.Get(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	translations, err := h.svc.Translations(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	resp := dto.FromModel(c)
	if len(translations) > 0 {
		resp.Translations = map[string]string{}
		for _, t := range translations {
			resp.Translations[t.Locale] = t.Name
		}
	}
	infrastructure.WriteJSON(w, http.StatusOK, resp)
}
