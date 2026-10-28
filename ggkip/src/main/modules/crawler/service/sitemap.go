package service

import (
	"encoding/xml"
	"io"
	"strings"
	"time"
)

type SitemapEntry struct {
	Loc     string
	LastMod time.Time
}

// ParseSitemap reads a <urlset> or a <sitemapindex>. Page URLs come back in
// urls, nested sitemap URLs in children.
func ParseSitemap(r io.Reader) (urls []SitemapEntry, children []string, err error) {
	var doc struct {
		URLs []struct {
			Loc     string `xml:"loc"`
			LastMod string `xml:"lastmod"`
		} `xml:"url"`
		Sitemaps []struct {
			Loc string `xml:"loc"`
		} `xml:"sitemap"`
	}
	dec := xml.NewDecoder(r)
	dec.Strict = false
	dec.CharsetReader = func(_ string, in io.Reader) (io.Reader, error) { return in, nil }
	if err := dec.Decode(&doc); err != nil {
		return nil, nil, err
	}
	for _, u := range doc.URLs {
		if loc := strings.TrimSpace(u.Loc); loc != "" {
			urls = append(urls, SitemapEntry{Loc: loc, LastMod: parseLastMod(u.LastMod)})
		}
	}
	for _, s := range doc.Sitemaps {
		if loc := strings.TrimSpace(s.Loc); loc != "" {
			children = append(children, loc)
		}
	}
	return urls, children, nil
}

func parseLastMod(v string) time.Time {
	v = strings.TrimSpace(v)
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04Z07:00", "2006-01-02T15:04:05", time.DateOnly} {
		if t, err := time.Parse(layout, v); err == nil {
			return t
		}
	}
	return time.Time{}
}
