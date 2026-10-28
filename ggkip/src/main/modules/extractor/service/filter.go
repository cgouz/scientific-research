package service

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"ggkip/src/main/modules/extractor/model"
)

// Filter reasons.
const (
	ReasonEmpty    = "empty"
	ReasonTooShort = "too_short"
	ReasonFewWords = "too_few_words"
	ReasonNoIndex  = "noindex"
)

// Filter decides whether extracted text is worth sending to the LLM.
type Filter struct {
	MinChars int
	MinWords int
}

// Check returns ok=false and a reason when the content should be skipped.
func (f Filter) Check(c model.Content) (ok bool, reason string) {
	switch {
	case c.NoIndex:
		return false, ReasonNoIndex
	case strings.TrimSpace(c.Text) == "":
		return false, ReasonEmpty
	case utf8.RuneCountInString(c.Text) < f.MinChars:
		return false, ReasonTooShort
	case WordCount(c.Text) < f.MinWords:
		return false, ReasonFewWords
	}
	return true, ""
}

// WordCount counts space-separated words, and each Han, Hiragana, Katakana
// or Hangul character as one word, since Japanese and Chinese text has no spaces.
func WordCount(s string) int {
	n, inWord := 0, false
	for _, r := range s {
		switch {
		case unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul):
			n++
			inWord = false
		case unicode.IsSpace(r) || unicode.IsPunct(r):
			inWord = false
		default:
			if !inWord {
				n++
				inWord = true
			}
		}
	}
	return n
}
