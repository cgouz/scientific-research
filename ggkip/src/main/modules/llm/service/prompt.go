package service

import (
	"ggkip/src/main/modules/llm/model"
)

const systemPrompt = `You normalize text from official government websites into one JSON object.

The user message is JSON with: the known country and organization (do not change or guess them),
the source URL, the page title, the clean page text, the allowed categories with their
subcategories, and the allowed record types.

Return ONLY a JSON object with these fields:
- record_type: exactly one of the allowed record types.
- category_code: exactly one allowed top-level category code. Use "UNKNOWN" if none fits; never invent codes.
- subcategory_code: a subcategory code that belongs to the chosen category, or null if none fits.
  It must be null when category_code is "UNKNOWN".
- title: a short, factual title in the language of the page.
- language: ISO 639-1 code of the page text (for example "en", "uz", "ru", "ja").
- published_at: publication date if the text states it, as YYYY-MM-DD or RFC 3339; otherwise null.
- content: the main content of the page as clean plain text in the original language. Keep facts,
  numbers, dates and names. Remove menus, boilerplate and repeated fragments. Do not translate.
- summary: one to three sentences summarizing the content, in the language of the page.
- keywords: 3 to 10 short keywords in the language of the page.
- confidence: a number from 0 to 1 for how well the category and subcategory fit.

Choose categories by their descriptions. Prefer a lower confidence over a wrong category.`

// OutputSchema is the JSON schema for NormalizeOutput, with enums for every
// code so a grammar-constrained runtime cannot produce unknown values.
func OutputSchema(t *model.Taxonomy) map[string]any {
	subcategory := map[string]any{"type": []string{"string", "null"}}
	if codes := t.SubcategoryCodes(); len(codes) > 0 {
		enum := make([]any, 0, len(codes)+1)
		for _, c := range codes {
			enum = append(enum, c)
		}
		subcategory["enum"] = append(enum, nil)
	}
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"record_type":      map[string]any{"type": "string", "enum": t.RecordTypes},
			"category_code":    map[string]any{"type": "string", "enum": t.CategoryCodes()},
			"subcategory_code": subcategory,
			"title":            map[string]any{"type": "string"},
			"language":         map[string]any{"type": "string"},
			"published_at":     map[string]any{"type": []string{"string", "null"}},
			"content":          map[string]any{"type": "string"},
			"summary":          map[string]any{"type": "string"},
			"keywords":         map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"confidence":       map[string]any{"type": "number", "minimum": 0, "maximum": 1},
		},
		"required": []string{"record_type", "category_code", "subcategory_code", "title", "language",
			"published_at", "content", "summary", "keywords", "confidence"},
		"additionalProperties": false,
	}
}
