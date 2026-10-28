package infrastructure

import (
	"bytes"
	"errors"
	"log/slog"
	"runtime"
	"strings"
	"testing"
)

func TestProtectReturnsPanicWithStack(t *testing.T) {
	deferRan := false
	err := Protect(func() {
		defer func() { deferRan = true }()
		panic("boom")
	})
	var p *PanicError
	if !errors.As(err, &p) || p.Value != "boom" || err.Error() != "panic: boom" {
		t.Fatalf("err = %v, want *PanicError for boom", err)
	}
	if !strings.Contains(string(p.Stack), "TestProtectReturnsPanicWithStack") {
		t.Error("the stack must point at the code that panicked")
	}
	if !deferRan {
		t.Error("deferred calls inside fn must still run")
	}
	if err := Protect(func() {}); err != nil {
		t.Errorf("no panic must give nil, got %v", err)
	}
}

func TestProtectRuntimeErrorsAndNilPanic(t *testing.T) {
	err := Protect(func() {
		var m map[string]int
		m["x"] = 1 // assignment to nil map
	})
	var re runtime.Error
	if !errors.As(err, &re) {
		t.Fatalf("a runtime panic must unwrap to runtime.Error, got %T %v", err, err)
	}

	err = Protect(func() { panic(nil) }) // Go 1.21+: recover() returns *runtime.PanicNilError
	if err == nil {
		t.Fatal("panic(nil) must not be reported as success")
	}
}

func TestRecoverAsErrorKeepsNamedResults(t *testing.T) {
	work := func() (n int, err error) {
		defer RecoverAsError(&err)
		n = 7
		panic(errors.New("disk on fire"))
	}
	n, err := work()
	if n != 7 || err == nil || !strings.Contains(err.Error(), "disk on fire") {
		t.Fatalf("n=%d err=%v", n, err)
	}
}

func TestLogPanicIncludesStack(t *testing.T) {
	var buf bytes.Buffer
	log := slog.New(slog.NewTextHandler(&buf, nil))
	LogPanic(log, "worker panicked", Protect(func() { panic("x") }), "url", "https://a.example/")
	out := buf.String()
	for _, want := range []string{"worker panicked", "url=https://a.example/", "stack=", "panic: x"} {
		if !strings.Contains(out, want) {
			t.Errorf("log %q misses %q", out, want)
		}
	}

	buf.Reset()
	LogPanic(log, "plain", errors.New("e"))
	if strings.Contains(buf.String(), "stack=") {
		t.Error("an ordinary error has no stack")
	}
}
