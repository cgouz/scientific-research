package handler

import (
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/category/dto"
	"ggkip/src/main/modules/category/model"
	"ggkip/src/main/modules/category/service"
)

type Handler struct {
	svc *service.Service
	log *slog.Logger
}

func New(svc *service.Service, log *slog.Logger) *Handler {
	return &Handler{svc: svc, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/categories", h.list)
	mux.HandleFunc("GET /v1/categories/{id}", h.get)
}

// list supports ?parent_id=, ?is_active=, ?top_level=true and ?tree=true
// (categories with their subcategories nested).
func (h *Handler) list(w http.ResponseWriter, r *http.Request) {
	parentID, err := infrastructure.QueryInt64(r, "parent_id")
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	active, err := infrastructure.QueryBool(r, "is_active")
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	topLevel, err := infrastructure.QueryBool(r, "top_level")
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	tree, err := infrastructure.QueryBool(r, "tree")
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	f := model.ListFilter{ParentID: parentID, IsActive: active, TopLevel: topLevel != nil && *topLevel}
	if tree != nil && *tree {
		f = model.ListFilter{IsActive: active}
	}
	items, err := h.svc.List(r.Context(), f)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}

	var out []dto.CategoryResponse
	if tree != nil && *tree {
		out = dto.Tree(items)
	} else {
		for _, c := range items {
			out = append(out, dto.FromModel(c))
		}
	}
	if out == nil {
		out = []dto.CategoryResponse{}
	}
	infrastructure.WriteJSON(w, http.StatusOK, infrastructure.ListResponse[dto.CategoryResponse]{Items: out})
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
	infrastructure.WriteJSON(w, http.StatusOK, dto.FromModel(c))
}
