package handler

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawl/dto"
	"ggkip/src/main/modules/crawl/model"
	"ggkip/src/main/modules/crawl/service"
)

type Handler struct {
	svc     *service.Service
	batches *service.BatchService
	log     *slog.Logger
}

func New(svc *service.Service, batches *service.BatchService, log *slog.Logger) *Handler {
	return &Handler{svc: svc, batches: batches, log: log}
}

func (h *Handler) Register(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/sites/{id}/crawl", h.trigger)
	mux.HandleFunc("GET /v1/crawl-runs", h.list)
	mux.HandleFunc("GET /v1/crawl-runs/{id}", h.get)
	mux.HandleFunc("POST /v1/crawl-batches", h.startBatch)
	mux.HandleFunc("GET /v1/crawl-batches", h.listBatches)
	mux.HandleFunc("GET /v1/crawl-batches/{id}", h.getBatch)
	mux.HandleFunc("POST /v1/crawl-batches/{id}/cancel", h.cancelBatch)
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

// list supports ?site_id=, ?batch_id=, ?status=, ?cursor=, ?limit= (newest first).
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
	if f.BatchID, err = infrastructure.QueryInt64(r, "batch_id"); err != nil {
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

// startBatch crawls every active official site of the active countries (or of
// the body's country_ids) in the background and returns 202 with the batch;
// follow it with GET /v1/crawl-batches/{id}. 409 while another batch runs.
func (h *Handler) startBatch(w http.ResponseWriter, r *http.Request) {
	var req dto.StartBatchRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&req); err != nil && !errors.Is(err, io.EOF) { // an empty body means all countries
		infrastructure.WriteError(w, h.log, fmt.Errorf("%w: body: %v", infrastructure.ErrInvalidInput, err))
		return
	}
	for _, id := range req.CountryIDs {
		if id <= 0 {
			infrastructure.WriteError(w, h.log, fmt.Errorf("%w: country_ids must be positive integers", infrastructure.ErrInvalidInput))
			return
		}
	}
	batch, err := h.batches.Start(r.Context(), req.CountryIDs)
	switch {
	case errors.Is(err, service.ErrBatchRunning):
		infrastructure.WriteJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
	case err != nil:
		infrastructure.WriteError(w, h.log, err)
	default:
		infrastructure.WriteJSON(w, http.StatusAccepted, dto.BatchFromModel(batch))
	}
}

// listBatches supports ?status=, ?cursor=, ?limit= (newest first).
func (h *Handler) listBatches(w http.ResponseWriter, r *http.Request) {
	page, err := infrastructure.ParsePage(r, 50, 500)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	items, err := h.batches.List(r.Context(), model.BatchListFilter{
		Status: infrastructure.QueryString(r, "status"), Cursor: page.Cursor, Limit: page.Limit,
	})
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	out := make([]dto.BatchResponse, len(items))
	for i, b := range items {
		out[i] = dto.BatchFromModel(b)
	}
	infrastructure.WriteJSON(w, http.StatusOK,
		infrastructure.NewListResponse(out, page.Limit, func(b dto.BatchResponse) int64 { return b.ID }))
}

// getBatch returns the batch with its crawl runs counted per status.
func (h *Handler) getBatch(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	batch, counts, err := h.batches.Get(r.Context(), id)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	infrastructure.WriteJSON(w, http.StatusOK, dto.BatchDetail(batch, counts))
}

// cancelBatch stops a running batch (202); 409 when it has already finished.
func (h *Handler) cancelBatch(w http.ResponseWriter, r *http.Request) {
	id, err := infrastructure.PathID(r)
	if err != nil {
		infrastructure.WriteError(w, h.log, err)
		return
	}
	batch, err := h.batches.Cancel(r.Context(), id)
	switch {
	case errors.Is(err, service.ErrBatchNotRunning):
		infrastructure.WriteJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
	case err != nil:
		infrastructure.WriteError(w, h.log, err)
	default:
		infrastructure.WriteJSON(w, http.StatusAccepted, dto.BatchFromModel(batch))
	}
}
