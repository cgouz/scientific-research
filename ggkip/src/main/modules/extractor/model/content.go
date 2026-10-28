package model

// Content is the clean textual form of a page. Text is what gets hashed and
// sent to the LLM; raw HTML never leaves the extractor.
type Content struct {
	Title    string   `json:"title"`
	Headings []string `json:"headings"`
	Text     string   `json:"text"`
	Language string   `json:"language,omitempty"`
	// Links are raw hrefs from the whole document (including navigation), for discovery.
	Links []string `json:"-"`
	// NoIndex / NoFollow come from <meta name="robots">.
	NoIndex  bool `json:"-"`
	NoFollow bool `json:"-"`
}
