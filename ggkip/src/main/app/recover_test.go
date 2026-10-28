package app

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sitemodel "ggkip/src/main/modules/site/model"
)

func TestWithRecoveryTurnsPanicInto500(t *testing.T) {
	var logBuf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&logBuf, nil))
	h := WithLogging(WithRecovery(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var m map[string]int
		m["boom"]++ // nil map write
	}), log), log)

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/records", nil))
	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), `"internal error"`) {
		t.Fatalf("got %d %q, want a 500 JSON reply", rec.Code, rec.Body.String())
	}
	out := logBuf.String()
	for _, want := range []string{"http handler panicked", "path=/v1/records", "stack=", "status=500"} {
		if !strings.Contains(out, want) {
			t.Errorf("log misses %q:\n%s", want, out)
		}
	}
}

func TestWithRecoveryKeepsAStartedResponse(t *testing.T) {
	h := WithRecovery(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, `{"items":[`)
		panic("halfway")
	}), quietLog)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusOK || rec.Body.String() != `{"items":[` {
		t.Fatalf("a second status line must not be written: %d %q", rec.Code, rec.Body.String())
	}
}

func TestWithRecoveryRepanicsAbortHandler(t *testing.T) {
	h := WithRecovery(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic(http.ErrAbortHandler)
	}), quietLog)
	defer func() {
		if v := recover(); v != http.ErrAbortHandler {
			t.Fatalf("recovered %v, want http.ErrAbortHandler passed on to net/http", v)
		}
	}()
	h.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodGet, "/", nil))
	t.Fatal("ErrAbortHandler must not be swallowed")
}

// panickingDue panics on its first call, then returns no sites.
type panickingDue struct{ calls atomic.Int32 }

func (p *panickingDue) Due(context.Context, int) ([]sitemodel.Site, error) {
	if p.calls.Add(1) == 1 {
		panic("bad row")
	}
	return nil, nil
}

func TestSchedulerKeepsRunningAfterPanickingTick(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	due := &panickingDue{}
	done := make(chan struct{})
	go func() {
		NewScheduler(due, &fakeStarter{capacity: 1}, 5*time.Millisecond, 1, quietLog).Run(ctx)
		close(done)
	}()
	deadline := time.After(2 * time.Second)
	for due.calls.Load() < 3 {
		select {
		case <-deadline:
			t.Fatalf("scheduler stopped after the panicking tick (%d ticks)", due.calls.Load())
		case <-time.After(5 * time.Millisecond):
		}
	}
	cancel()
	<-done
}
