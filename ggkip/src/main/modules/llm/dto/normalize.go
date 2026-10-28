package dto

// NormalizeInput is what the LLM receives. Country and organization are
// already known from the site; the LLM never has to guess them.
type NormalizeInput struct {
	Country      NamedRef           `json:"country"`
	Organization NamedRef           `json:"organization"`
	SourceURL    string             `json:"source_url"`
	PageTitle    string             `json:"page_title"`
	Content      string             `json:"content"`
	Categories   []TaxonomyCategory `json:"categories"`
	RecordTypes  []string           `json:"record_types"`
	// OmitContent asks for an empty content field; set after a reply was cut
	// off at max_tokens. The extracted page text is stored instead.
	OmitContent bool `json:"omit_content,omitempty"`
}

type NamedRef struct {
	ID   int64  `json:"id"`
	Name string `json:"name"`
}

type TaxonomyCategory struct {
	Code          string             `json:"code"`
	Name          string             `json:"name"`
	Description   string             `json:"description"`
	Subcategories []TaxonomyCategory `json:"subcategories,omitempty"`
}

// NormalizeOutput is the strict JSON the LLM must return.
type NormalizeOutput struct {
	RecordType      string   `json:"record_type"`
	CategoryCode    string   `json:"category_code"`
	SubcategoryCode *string  `json:"subcategory_code"`
	Title           string   `json:"title"`
	Language        string   `json:"language"`
	PublishedAt     *string  `json:"published_at"`
	Content         string   `json:"content"`
	Summary         string   `json:"summary"`
	Keywords        []string `json:"keywords"`
	Confidence      *float64 `json:"confidence"`
}
