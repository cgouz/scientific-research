package infrastructure

import (
	"fmt"
	"log/slog"
	"runtime/debug"
)

// PanicError is a recovered panic together with the stack of the goroutine
// that panicked, so a crash in one page, site or request becomes an ordinary
// error instead of stopping the whole process.
type PanicError struct {
	Value any
	Stack []byte
}

func (e *PanicError) Error() string { return fmt.Sprintf("panic: %v", e.Value) }

// Unwrap returns the panic value when it is an error (runtime errors such as
// a nil map write or an index out of range are), so errors.Is/As still work.
func (e *PanicError) Unwrap() error {
	err, _ := e.Value.(error)
	return err
}

// RecoverAsError turns a panic in the calling function into a *PanicError
// stored in *errp. It must be deferred directly:
//
//	func work() (err error) {
//		defer infrastructure.RecoverAsError(&err)
//		...
//	}
func RecoverAsError(errp *error) {
	if v := recover(); v != nil {
		*errp = &PanicError{Value: v, Stack: debug.Stack()}
	}
}

// Protect runs fn and returns its panic as a *PanicError (nil if fn returned
// normally). Deferred calls inside fn still run before Protect returns.
func Protect(fn func()) (err error) {
	defer RecoverAsError(&err)
	fn()
	return nil
}

// LogPanic logs err at ERROR level; for a *PanicError it adds the stack.
func LogPanic(log *slog.Logger, msg string, err error, attrs ...any) {
	if p, ok := err.(*PanicError); ok {
		attrs = append(attrs, "stack", string(p.Stack))
	}
	log.Error(msg, append(attrs, "error", err)...)
}
