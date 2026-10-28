package infrastructure

import (
	"context"
	"database/sql"
	"fmt"
	"io/fs"
	"log/slog"
	"sort"
	"strconv"
	"strings"
)

// migrationLockKey is an arbitrary constant for pg_advisory_lock, so two
// instances starting together do not apply the same migration twice.
const migrationLockKey = 727274001

type migration struct {
	Version  int64
	Filename string
}

// Migrate applies every NNN_name.sql file in fsys that is not yet recorded in
// schema_migrations, each in its own transaction, in version order.
func Migrate(ctx context.Context, db *sql.DB, fsys fs.FS, log *slog.Logger) (int, error) {
	migrations, err := listMigrations(fsys)
	if err != nil {
		return 0, err
	}

	conn, err := db.Conn(ctx)
	if err != nil {
		return 0, err
	}
	defer conn.Close()

	if _, err := conn.ExecContext(ctx, `SELECT pg_advisory_lock($1)`, migrationLockKey); err != nil {
		return 0, fmt.Errorf("migration lock: %w", err)
	}
	defer conn.ExecContext(context.WithoutCancel(ctx), `SELECT pg_advisory_unlock($1)`, migrationLockKey)

	if _, err := conn.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version     BIGINT PRIMARY KEY,
			filename    TEXT NOT NULL,
			applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)`); err != nil {
		return 0, fmt.Errorf("create schema_migrations: %w", err)
	}

	applied := map[int64]bool{}
	rows, err := conn.QueryContext(ctx, `SELECT version FROM schema_migrations`)
	if err != nil {
		return 0, err
	}
	for rows.Next() {
		var v int64
		if err := rows.Scan(&v); err != nil {
			rows.Close()
			return 0, err
		}
		applied[v] = true
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}

	count := 0
	for _, m := range migrations {
		if applied[m.Version] {
			continue
		}
		body, err := fs.ReadFile(fsys, m.Filename)
		if err != nil {
			return count, err
		}
		tx, err := conn.BeginTx(ctx, nil)
		if err != nil {
			return count, err
		}
		if _, err := tx.ExecContext(ctx, string(body)); err != nil {
			tx.Rollback()
			return count, fmt.Errorf("migration %s: %w", m.Filename, err)
		}
		if _, err := tx.ExecContext(ctx,
			`INSERT INTO schema_migrations (version, filename) VALUES ($1, $2)`, m.Version, m.Filename); err != nil {
			tx.Rollback()
			return count, fmt.Errorf("record migration %s: %w", m.Filename, err)
		}
		if err := tx.Commit(); err != nil {
			return count, fmt.Errorf("commit migration %s: %w", m.Filename, err)
		}
		log.Info("migration applied", "file", m.Filename)
		count++
	}
	return count, nil
}

func listMigrations(fsys fs.FS) ([]migration, error) {
	entries, err := fs.ReadDir(fsys, ".")
	if err != nil {
		return nil, fmt.Errorf("read migrations: %w", err)
	}
	var out []migration
	seen := map[int64]string{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".sql") {
			continue
		}
		prefix, _, ok := strings.Cut(name, "_")
		version, err := strconv.ParseInt(prefix, 10, 64)
		if !ok || err != nil || version <= 0 {
			return nil, fmt.Errorf("migration %s: name must look like 001_description.sql", name)
		}
		if other, dup := seen[version]; dup {
			return nil, fmt.Errorf("migrations %s and %s share version %d", other, name, version)
		}
		seen[version] = name
		out = append(out, migration{Version: version, Filename: name})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version < out[j].Version })
	return out, nil
}
