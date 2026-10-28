package repo

import (
	"strings"
	"testing"
)

func TestSiteColumnsPrefixesEveryColumn(t *testing.T) {
	got := siteColumns("s.")
	names := strings.Split(got, ", ")
	if len(names) != len(strings.Split(columns, ",")) {
		t.Fatalf("column count changed: %q", got)
	}
	for _, n := range names {
		if !strings.HasPrefix(n, "s.") || strings.ContainsAny(n, " \t\n") {
			t.Errorf("bad column %q in %q", n, got)
		}
	}
}
