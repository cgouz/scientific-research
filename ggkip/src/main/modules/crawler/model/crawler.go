package model

import "time"

// Settings come from resources/settings/crawler.json.
type Settings struct {
	MaxPagesPerRun        int      `json:"max_pages_per_run"`
	MaxDepth              int      `json:"max_depth"`
	MaxSitemaps           int      `json:"max_sitemaps"`
	MaxRedirects          int      `json:"max_redirects"`
	MaxRetries            int      `json:"max_retries"`
	RequestsPerSecond     float64  `json:"requests_per_second"`
	MinTextChars          int      `json:"min_text_chars"`
	MinTextWords          int      `json:"min_text_words"`
	AllowedContentTypes   []string `json:"allowed_content_types"`
	SkipExtensions        []string `json:"skip_extensions"`
	TrackingParams        []string `json:"tracking_params"`
	TrackingParamPrefixes []string `json:"tracking_param_prefixes"`
}

// Options come from environment configuration.
type Options struct {
	Workers      int
	Timeout      time.Duration
	MaxBodyBytes int64
	UserAgent    string
}

// Target is the site being crawled.
type Target struct {
	Domain  string // e.g. "gov.uk"; the crawl stays on Domain and www.Domain
	BaseURL string
}

// FetchResult is handed to the crawl handler for every URL the crawler tried.
// Exactly one of Body (success), SkipReason or Err describes the outcome.
type FetchResult struct {
	URL         string // normalized URL that was requested
	FinalURL    string // normalized URL after redirects
	Depth       int
	StatusCode  int // 0 when no response was received
	ContentType string
	Body        []byte // UTF-8
	SkipReason  string
	Err         error
	Duration    time.Duration
}

// Skip reasons.
const (
	SkipContentType = "content_type"
	SkipTooLarge    = "too_large"
	SkipOutOfScope  = "redirect_out_of_scope"
)

// Stats summarize what the crawler itself did (the handler counts the rest).
type Stats struct {
	URLsQueued   int
	SitemapURLs  int
	RobotsLoaded bool
}
