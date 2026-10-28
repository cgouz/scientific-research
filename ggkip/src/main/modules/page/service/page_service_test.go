package service

import (
	"testing"

	"ggkip/src/main/infrastructure"
)

func TestClassify(t *testing.T) {
	a := infrastructure.SHA256("clean text v1")
	b := infrastructure.SHA256("clean text v2")
	if got := Classify(nil, a); got != ChangeNew {
		t.Errorf("never processed page = %v, want new", got)
	}
	if got := Classify(a, infrastructure.SHA256("clean text v1")); got != ChangeUnchanged {
		t.Errorf("same hash = %v, want unchanged (no LLM call)", got)
	}
	if got := Classify(a, b); got != ChangeChanged {
		t.Errorf("different hash = %v, want changed", got)
	}
	if ChangeNew.Status() != "new" || ChangeChanged.Status() != "changed" || ChangeUnchanged.Status() != "unchanged" {
		t.Error("unexpected status names")
	}
}
