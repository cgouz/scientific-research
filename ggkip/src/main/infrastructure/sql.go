package infrastructure

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

// Where builds a parameterized WHERE clause. Each condition uses "?" for its
// single value, which is rewritten to the next $n placeholder.
type Where struct {
	conds []string
	Args  []any
}

func (w *Where) Add(cond string, value any) {
	w.Args = append(w.Args, value)
	w.conds = append(w.conds, strings.Replace(cond, "?", "$"+strconv.Itoa(len(w.Args)), 1))
}

// Raw adds a condition that takes no value, such as "parent_id IS NULL".
func (w *Where) Raw(cond string) {
	w.conds = append(w.conds, cond)
}

// Arg appends a value and returns its placeholder, for LIMIT and similar.
func (w *Where) Arg(value any) string {
	w.Args = append(w.Args, value)
	return "$" + strconv.Itoa(len(w.Args))
}

func (w *Where) SQL() string {
	if len(w.conds) == 0 {
		return ""
	}
	return " WHERE " + strings.Join(w.conds, " AND ")
}

// ResetIdentity moves a table's identity sequence past the highest id, so
// inserts without an explicit id do not collide with seeded ids.
func ResetIdentity(ctx context.Context, db DBTX, table string) error {
	_, err := db.ExecContext(ctx, fmt.Sprintf(
		`SELECT setval(pg_get_serial_sequence('%[1]s', 'id'), GREATEST((SELECT MAX(id) FROM %[1]s), 1))`, table))
	if err != nil {
		return fmt.Errorf("reset %s identity: %w", table, err)
	}
	return nil
}
