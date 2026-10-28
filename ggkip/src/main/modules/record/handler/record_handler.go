package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/record/dto"
	"ggkip/src/main/modules/record/model"
	"ggkip/src/main/modules/record/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/records", h.list)
	mux.HandleFunc("GET /v1/records/{id}", h.get)
}

// list supports ?country_id=, ?organization_id=, ?site_id=, ?category_id=,
// ?subcategory_id=, ?record_type=, ?language=, ?published_from=,
// ?published_to=, ?cursor=, ?limit= (newest first).
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
	out := make([]dto.RecordSummary, len(items))
	for i, rec := range items {
		out[i] = dto.SummaryFromModel(rec)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, f.Limit, func(s dto.RecordSummary) int64 { return s.ID }))
}

func parseFilter(r *http.Request) (model.ListFilter, error) {
	page, err := infrastructure.ParsePage(r, 50, 500)
	if err != nil {
		return model.ListFilter{}, err
	}
	f := model.ListFilter{
		RecordType: infrastructure.QueryString(r, "record_type"),
		Language:   infrastructure.QueryString(r, "language"),
		Cursor:     page.Cursor,
		Limit:      page.Limit,
	}
	for key, dst := range map[string]**int64{
		"country_id": &f.CountryID, "organization_id": &f.OrganizationID, "site_id": &f.SiteID,
		"category_id": &f.CategoryID, "subcategory_id": &f.SubcategoryID,
	} {
		if *dst, err = infrastructure.QueryInt64(r, key); err != nil {
			return f, err
		}
	}
	if f.PublishedFrom, err = infrastructure.QueryTime(r, "published_from"); err != nil {
		return f, err
	}
	if f.PublishedTo, err = infrastructure.QueryTime(r, "published_to"); err != nil {
		return f, err
	}
	return f, nil
}

func (h *Handler) get(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	rec, err := h.svc.Get(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	infrastructure.WriteJSON(w, http.StatusOK, dto.FromModel(rec))
}
