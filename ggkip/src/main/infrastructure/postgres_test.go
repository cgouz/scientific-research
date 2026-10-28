package infrastructure

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"sync/atomic"
	"testing"
)

// fakeDriver counts transactions so InTx can be tested without PostgreSQL.
type fakeDriver struct{ begun, committed, rolledBack atomic.Int32 }

func (d *fakeDriver) Open(string) (driver.Conn, error) { return fakeConn{d}, nil }

type fakeConn struct{ d *fakeDriver }

func (c fakeConn) Prepare(string) (driver.Stmt, error) { return nil, errors.New("not supported") }
func (c fakeConn) Close() error                        { return nil }
func (c fakeConn) Begin() (driver.Tx, error)           { c.d.begun.Add(1); return fakeTx(c), nil }

type fakeTx struct{ d *fakeDriver }

func (t fakeTx) Commit() error   { t.d.committed.Add(1); return nil }
func (t fakeTx) Rollback() error { t.d.rolledBack.Add(1); return nil }

func TestInTxRollsBackOnErrorAndPanic(t *testing.T) {
	d := &fakeDriver{}
	sql.Register("fake-intx", d)
	db, err := sql.Open("fake-intx", "")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1) // a leaked transaction would block every later call
	defer db.Close()
	ctx := context.Background()

	if err := InTx(ctx, db, func(*sql.Tx) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if err := InTx(ctx, db, func(*sql.Tx) error { return errors.New("boom") }); err == nil {
		t.Fatal("fn error must be returned")
	}
	if p := Protect(func() { InTx(ctx, db, func(*sql.Tx) error { panic("bug") }) }); p == nil {
		t.Fatal("the panic must reach the caller")
	}
	// The single connection must be free again.
	if err := InTx(ctx, db, func(*sql.Tx) error { return nil }); err != nil {
		t.Fatal(err)
	}
	if d.begun.Load() != 4 || d.committed.Load() != 2 || d.rolledBack.Load() != 2 {
		t.Fatalf("begun=%d committed=%d rolled back=%d, want 4/2/2",
			d.begun.Load(), d.committed.Load(), d.rolledBack.Load())
	}
}
