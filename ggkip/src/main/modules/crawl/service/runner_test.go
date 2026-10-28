package service

import (
	"context"
	"errors"
	"testing"
	"time"

	crawlermodel "ggkip/src/main/modules/crawler/model"
)

// A runner without a database: enough for slot handling and the panic guards
// (any database call panics on the nil *sql.DB, which is what we want here).
func testRunner(maxSites int) *Runner {
	return NewRunner(context.Background(), RunnerDeps{MaxSites: maxSites, LLMWorkers: 1, Log: quietLog})
}

func TestAcquireWithoutWait(t *testing.T) {
	r := testRunner(1)
	release, err := r.acquire(context.Background(), 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.acquire(context.Background(), 1, false); !errors.Is(err, ErrAlreadyRunning) {
		t.Errorf("same site = %v, want ErrAlreadyRunning", err)
	}
	if _, err := r.acquire(context.Background(), 2, false); !errors.Is(err, ErrBusy) {
		t.Errorf("full runner = %v, want ErrBusy", err)
	}
	release()
	release2, err := r.acquire(context.Background(), 2, false)
	if err != nil {
		t.Fatalf("after release the slot is free: %v", err)
	}
	release2()
}

func TestAcquireWaitsForASlot(t *testing.T) {
	r := testRunner(1)
	release, _ := r.acquire(context.Background(), 1, false)
	got := make(chan error, 1)
	go func() {
		rel, err := r.acquire(context.Background(), 2, true)
		if err == nil {
			rel()
		}
		got <- err
	}()
	select {
	case err := <-got:
		t.Fatalf("acquire returned %v before a slot was free", err)
	case <-time.After(20 * time.Millisecond):
	}
	release()
	select {
	case err := <-got:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("a waiting site must get the freed slot")
	}
}

func TestAcquireCancelledWhileWaitingLeaksNothing(t *testing.T) {
	r := testRunner(1)
	release, _ := r.acquire(context.Background(), 1, false)
	ctx, cancel := context.WithCancel(context.Background())
	got := make(chan error, 1)
	go func() {
		_, err := r.acquire(ctx, 2, true)
		got <- err
	}()
	time.Sleep(10 * time.Millisecond)
	cancel()
	if err := <-got; !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	release()
	// site 2 is no longer marked and the only slot is free again
	rel, err := r.acquire(context.Background(), 2, false)
	if err != nil {
		t.Fatalf("a cancelled wait must leave no mark and take no slot: %v", err)
	}
	rel()

	// already cancelled: never takes a slot, even though one is free
	if _, err := r.acquire(ctx, 3, true); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if rel, err := r.acquire(context.Background(), 4, false); err != nil {
		t.Fatalf("slot leaked by a cancelled acquire: %v", err)
	} else {
		rel()
	}
}

func TestPageAndJobPanicsFailOnlyThatPage(t *testing.T) {
	r := testRunner(1)
	rn := &run{}

	// handlePage stores the page through the (nil) database and panics.
	links := r.handlePageSafe(context.Background(), rn, &crawlermodel.FetchResult{
		URL: "https://a.example/x", FinalURL: "https://a.example/x", ContentType: "text/html",
		Body: []byte(`<html><body><a href="/y">y</a></body></html>`),
	}, quietLog)
	if links != nil || rn.counters.PagesFailed.Load() != 1 {
		t.Fatalf("links=%v failed=%d; a panicking page gives no links and counts as failed",
			links, rn.counters.PagesFailed.Load())
	}

	// normalizeAndStore calls the (nil) normalizer; marking the page failed
	// then panics on the nil database too, and the worker must survive both.
	r.processJob(context.Background(), rn, pageJob{url: "https://a.example/z"}, quietLog)
	if rn.counters.PagesFailed.Load() != 2 {
		t.Fatalf("failed = %d, want 2", rn.counters.PagesFailed.Load())
	}
}
