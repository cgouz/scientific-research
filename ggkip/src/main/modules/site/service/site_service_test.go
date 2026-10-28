package service

import (
	"testing"
	"time"
)

func TestNextCrawlAt(t *testing.T) {
	at := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	if got := NextCrawlAt(at, 6*time.Hour); !got.Equal(at.Add(6 * time.Hour)) {
		t.Errorf("NextCrawlAt = %v", got)
	}
}

func TestRetryAt(t *testing.T) {
	at := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	cases := []struct {
		interval, retry, want time.Duration
	}{
		{6 * time.Hour, 30 * time.Minute, 30 * time.Minute},    // normal retry delay
		{10 * time.Minute, 30 * time.Minute, 10 * time.Minute}, // never later than the regular interval
		{6 * time.Hour, 0, 6 * time.Hour},                      // no retry delay configured
	}
	for _, c := range cases {
		if got := RetryAt(at, c.interval, c.retry); !got.Equal(at.Add(c.want)) {
			t.Errorf("RetryAt(interval=%v, retry=%v) = +%v, want +%v", c.interval, c.retry, got.Sub(at), c.want)
		}
	}
}
