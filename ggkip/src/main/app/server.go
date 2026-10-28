package app

import (
	"context"
	"database/sql"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"ggkip/src/main/infrastructure"
)

// RegisterHealth adds GET /health, which also pings the database.
func RegisterHealth(mux *http.ServeMux, db *sql.DB, runtime string) {
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := db.PingContext(ctx); err != nil {
			infrastructure.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "down", "database": err.Error()})
			return
		}
		infrastructure.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok", "llm_runtime": runtime})
	})
}

// WithLogging logs every request at INFO with method, path, status and duration.
func WithLogging(next http.Handler, log *slog.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r)
		log.Info("http request", "method", r.Method, "path", r.URL.Path, "status", rec.status, "duration", time.Since(start))
	})
}

// WithRecovery turns a panic in a handler into a logged 500 JSON reply
// instead of a dropped connection. http.ErrAbortHandler is re-panicked: the
// net/http server uses it on purpose to abort a response.
func WithRecovery(next http.Handler, log *slog.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		err := infrastructure.Protect(func() { next.ServeHTTP(rec, r) })
		if err == nil {
			return
		}
		var p *infrastructure.PanicError
		if errors.As(err, &p) && p.Value == http.ErrAbortHandler {
			panic(http.ErrAbortHandler)
		}
		infrastructure.LogPanic(log, "http handler panicked", err, "method", r.Method, "path", r.URL.Path)
		if !rec.wrote { // otherwise the status line is already sent; the client sees a cut-off body
			infrastructure.WriteJSON(rec, http.StatusInternalServerError, map[string]string{"error": "internal error"})
		}
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	wrote  bool
}

func (r *statusRecorder) WriteHeader(code int) {
	if !r.wrote {
		r.status, r.wrote = code, true
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	r.wrote = true
	return r.ResponseWriter.Write(b)
}

// Unwrap lets http.ResponseController reach the real writer (flush, deadlines).
func (r *statusRecorder) Unwrap() http.ResponseWriter { return r.ResponseWriter }

// RunServer serves until ctx is cancelled, then shuts down gracefully.
func RunServer(ctx context.Context, addr string, handler http.Handler, log *slog.Logger) error {
	srv := &http.Server{
		Addr:              addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	errc := make(chan error, 1)
	go func() {
		log.Info("http server listening", "addr", addr)
		errc <- srv.ListenAndServe()
	}()
	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}
