package dto

import (
	"encoding/hex"
	"encoding/json"
	"time"

	"ggkip/src/main/modules/record/model"
)

// RecordSummary is a list item; the full content is only returned by GET /v1/records/{id}.
type RecordSummary struct {
	ID             int64      `json:"id"`
	CountryID      int64      `json:"country_id"`
	OrganizationID int64      `json:"organization_id"`
	SiteID         int64      `json:"site_id"`
	PageID         int64      `json:"page_id"`
	SourceURL      string     `json:"source_url"`
	CategoryID     int64      `json:"category_id"`
	SubcategoryID  *int64     `json:"subcategory_id"`
	RecordType     string     `json:"record_type"`
	Title          *string    `json:"title"`
	LanguageCode   *string    `json:"language_code"`
	PublishedAt    *time.Time `json:"published_at"`
	Summary        *string    `json:"summary"`
	Keywords       []string   `json:"keywords"`
	Confidence     *float64   `json:"confidence"`
	CreatedAt      time.Time  `json:"created_at"`
	UpdatedAt      time.Time  `json:"updated_at"`
}

type RecordResponse struct {
	RecordSummary
	Content     string          `json:"content"`
	Metadata    json.RawMessage `json:"metadata"`
	ContentHash string          `json:"content_hash"`
	LLMRuntime  *string         `json:"llm_runtime"`
	LLMModel    *string         `json:"llm_model"`
}

func SummaryFromModel(r model.Record) RecordSummary {
	return RecordSummary{
		ID: r.ID, CountryID: r.CountryID, OrganizationID: r.OrganizationID, SiteID: r.SiteID, PageID: r.PageID,
		SourceURL: r.SourceURL, CategoryID: r.CategoryID, SubcategoryID: r.SubcategoryID, RecordType: r.RecordType,
		Title: r.Title, LanguageCode: r.LanguageCode, PublishedAt: r.PublishedAt, Summary: r.Summary,
		Keywords: r.Keywords, Confidence: r.Confidence, CreatedAt: r.CreatedAt, UpdatedAt: r.UpdatedAt,
	}
}

func FromModel(r model.Record) RecordResponse {
	return RecordResponse{
		RecordSummary: SummaryFromModel(r),
		Content:       r.Content,
		Metadata:      r.Metadata,
		ContentHash:   hex.EncodeToString(r.ContentHash),
		LLMRuntime:    r.LLMRuntime,
		LLMModel:      r.LLMModel,
	}
}
