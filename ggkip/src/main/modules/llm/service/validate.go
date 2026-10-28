package service

import (
	"fmt"
	"math"
	"strings"
	"time"

	"ggkip/src/main/modules/llm/dto"
	"ggkip/src/main/modules/llm/model"
)

const (
	maxKeywords   = 20
	maxKeywordLen = 100
	maxLangLen    = 10
)

// Validate checks LLM output against the taxonomy and resolves codes to
// category ids. Only a valid result may be stored.
func Validate(out dto.NormalizeOutput, t *model.Taxonomy) (model.Result, error) {
	var problems []string
	res := model.Result{
		RecordType: strings.ToUpper(strings.TrimSpace(out.RecordType)),
		Title:      strings.TrimSpace(out.Title),
		Language:   strings.ToLower(strings.TrimSpace(out.Language)),
		Content:    strings.TrimSpace(out.Content),
		Summary:    strings.TrimSpace(out.Summary),
	}

	if !t.HasRecordType(res.RecordType) {
		problems = append(problems, fmt.Sprintf("record_type %q is not allowed", out.RecordType))
	}

	categoryCode := strings.TrimSpace(out.CategoryCode)
	category, ok := t.Lookup(categoryCode)
	switch {
	case !ok:
		problems = append(problems, fmt.Sprintf("category_code %q does not exist", out.CategoryCode))
	case category.ParentID != nil:
		problems = append(problems, fmt.Sprintf("category_code %q is a subcategory", out.CategoryCode))
	default:
		res.CategoryID = category.ID
	}

	if out.SubcategoryCode != nil && strings.TrimSpace(*out.SubcategoryCode) != "" {
		code := strings.TrimSpace(*out.SubcategoryCode)
		sub, ok := t.Lookup(code)
		switch {
		case !ok:
			problems = append(problems, fmt.Sprintf("subcategory_code %q does not exist", code))
		case sub.ParentID == nil:
			problems = append(problems, fmt.Sprintf("subcategory_code %q is a top-level category", code))
		case *sub.ParentID != category.ID:
			problems = append(problems, fmt.Sprintf("subcategory_code %q does not belong to %q", code, categoryCode))
		default:
			id := sub.ID
			res.SubcategoryID = &id
		}
	}

	switch {
	case out.Confidence == nil:
		problems = append(problems, "confidence is missing")
	case math.IsNaN(*out.Confidence) || *out.Confidence < 0 || *out.Confidence > 1:
		problems = append(problems, fmt.Sprintf("confidence %v is outside 0..1", *out.Confidence))
	default:
		res.Confidence = *out.Confidence
	}

	if out.PublishedAt != nil && strings.TrimSpace(*out.PublishedAt) != "" {
		if t, ok := parseDate(*out.PublishedAt); ok {
			res.PublishedAt = &t
		} else {
			problems = append(problems, fmt.Sprintf("published_at %q is not a date", *out.PublishedAt))
		}
	}

	if len(res.Language) > maxLangLen {
		problems = append(problems, fmt.Sprintf("language %q is too long", out.Language))
	}

	for _, k := range out.Keywords {
		k = strings.TrimSpace(k)
		if k != "" && len(k) <= maxKeywordLen && len(res.Keywords) < maxKeywords {
			res.Keywords = append(res.Keywords, k)
		}
	}

	if len(problems) > 0 {
		return res, &model.ValidationError{Problems: problems}
	}
	return res, nil
}

func parseDate(v string) (time.Time, bool) {
	v = strings.TrimSpace(v)
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05", time.DateOnly} {
		if t, err := time.Parse(layout, v); err == nil {
			if t.Year() < 1900 || t.After(time.Now().AddDate(1, 0, 0)) {
				return time.Time{}, false
			}
			return t, true
		}
	}
	return time.Time{}, false
}
