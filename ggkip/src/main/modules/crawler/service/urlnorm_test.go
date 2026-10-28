package service

import (
	"encoding/hex"
	"net/url"
	"testing"

	"ggkip/src/main/modules/crawler/model"
)

var testSettings = model.Settings{
	MaxPagesPerRun:        100,
	MaxDepth:              3,
	MaxSitemaps:           5,
	MaxRedirects:          5,
	MaxRetries:            0,
	RequestsPerSecond:     1000,
	AllowedContentTypes:   []string{"text/html", "text/plain", "application/json"},
	SkipExtensions:        []string{".pdf", ".docx", ".jpg", ".zip"},
	TrackingParams:        []string{"gclid", "fbclid"},
	TrackingParamPrefixes: []string{"utm_"},
}

func TestNormalize(t *testing.T) {
	n := NewURLNormalizer(testSettings)
	base, _ := url.Parse("https://www.example.gov/news/list/")

	cases := []struct {
		in, want string
	}{
		{"HTTPS://WWW.Example.GOV", "https://www.example.gov/"},
		{"https://example.gov:443/a", "https://example.gov/a"},
		{"http://example.gov:80/a", "http://example.gov/a"},
		{"http://example.gov:8080/a", "http://example.gov:8080/a"},
		{"https://example.gov/a#section", "https://example.gov/a"},
		{"https://example.gov/a/./b/../c", "https://example.gov/a/c"},
		{"https://example.gov//a//b/", "https://example.gov/a/b/"},
		{"https://example.gov/a/", "https://example.gov/a/"},
		{"https://example.gov/a?utm_source=x&id=7&gclid=1", "https://example.gov/a?id=7"},
		{"https://example.gov/a?b=2&a=1", "https://example.gov/a?a=1&b=2"},
		{"https://example.gov/a?utm_medium=x", "https://example.gov/a"},
		{"https://user:pw@example.gov/a", "https://example.gov/a"},
		{"item?id=5", "https://www.example.gov/news/list/item?id=5"},
		{"../archive", "https://www.example.gov/news/archive"},
		{"/about", "https://www.example.gov/about"},
		{"//example.gov/x", "https://example.gov/x"},
	}
	for _, c := range cases {
		got, err := n.Normalize(c.in, base)
		if err != nil {
			t.Errorf("Normalize(%q): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("Normalize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestNormalizeRejectsOtherSchemes(t *testing.T) {
	n := NewURLNormalizer(testSettings)
	for _, in := range []string{"mailto:a@example.gov", "javascript:void(0)", "ftp://example.gov/f", "tel:+1"} {
		if _, err := n.Normalize(in, nil); err == nil {
			t.Errorf("Normalize(%q) should fail", in)
		}
	}
}

func TestIsSkippable(t *testing.T) {
	n := NewURLNormalizer(testSettings)
	for in, want := range map[string]bool{
		"https://example.gov/report.pdf":      true,
		"https://example.gov/REPORT.PDF":      true,
		"https://example.gov/photo.jpg?w=100": true,
		"https://example.gov/archive.zip":     true,
		"https://example.gov/news/":           false,
		"https://example.gov/page.html":       false,
		"https://example.gov/page":            false,
		"https://example.gov/v1.2/item":       false,
	} {
		u, _ := url.Parse(in)
		if got := n.IsSkippable(u); got != want {
			t.Errorf("IsSkippable(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestScope(t *testing.T) {
	s := NewScope("www.Example.gov")
	for in, want := range map[string]bool{
		"https://example.gov/a":      true,
		"https://www.example.gov/a":  true,
		"https://EXAMPLE.GOV./a":     true,
		"https://news.example.gov/a": false,
		"https://example.gov.evil/a": false,
		"https://notexample.gov/a":   false,
		"https://example.gov:8443/a": true,
	} {
		u, _ := url.Parse(in)
		if got := s.Allows(u); got != want {
			t.Errorf("Allows(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestHashURL(t *testing.T) {
	a := hex.EncodeToString(HashURL("https://example.gov/"))
	if len(a) != 64 {
		t.Fatalf("hash length = %d, want 64 hex chars", len(a))
	}
	if a != hex.EncodeToString(HashURL("https://example.gov/")) {
		t.Fatal("hash is not deterministic")
	}
	if a == hex.EncodeToString(HashURL("https://example.gov/a")) {
		t.Fatal("different URLs must hash differently")
	}
}
