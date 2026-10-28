package service

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"ggkip/src/main/modules/llm/dto"
	"ggkip/src/main/modules/llm/model"
)

func ptr[T any](v T) *T { return &v }

func testTaxonomy(t *testing.T) *model.Taxonomy {
	t.Helper()
	tax, err := model.NewTaxonomy([]model.Entry{
		{ID: 1, Code: "FIN", Name: "Finance"},
		{ID: 2, Code: "HEALTH", Name: "Health"},
		{ID: 3, Code: "UNKNOWN", Name: "Unknown"},
		{ID: 10, ParentID: ptr(int64(1)), Code: "FIN.TAX", Name: "Taxation"},
		{ID: 11, ParentID: ptr(int64(1)), Code: "FIN.CUSTOMS", Name: "Customs"},
		{ID: 20, ParentID: ptr(int64(2)), Code: "HEALTH.MEDICINES", Name: "Medicines"},
	}, model.TaxonomySettings{UnknownCategoryCode: "UNKNOWN", RecordTypes: []string{"NEWS", "REGULATION", "OTHER"}})
	if err != nil {
		t.Fatal(err)
	}
	return tax
}

func validOutput() dto.NormalizeOutput {
	return dto.NormalizeOutput{
		RecordType: "REGULATION", CategoryCode: "FIN", SubcategoryCode: ptr("FIN.TAX"),
		Title: "Changes to VAT rates", Language: "en", PublishedAt: ptr("2026-01-01"),
		Content: "From January 1, VAT rates change.", Summary: "VAT rates change.",
		Keywords: []string{"VAT", " tax ", ""}, Confidence: ptr(0.98),
	}
}

func TestTaxonomyRequiresUnknown(t *testing.T) {
	_, err := model.NewTaxonomy([]model.Entry{{ID: 1, Code: "FIN"}},
		model.TaxonomySettings{UnknownCategoryCode: "UNKNOWN", RecordTypes: []string{"OTHER"}})
	if err == nil {
		t.Fatal("taxonomy without UNKNOWN must be rejected")
	}
}

func TestValidateAccepts(t *testing.T) {
	res, err := Validate(validOutput(), testTaxonomy(t))
	if err != nil {
		t.Fatal(err)
	}
	if res.CategoryID != 1 || res.SubcategoryID == nil || *res.SubcategoryID != 10 {
		t.Errorf("codes not resolved to ids: %+v", res)
	}
	if res.PublishedAt == nil || res.PublishedAt.Year() != 2026 {
		t.Errorf("published_at not parsed: %v", res.PublishedAt)
	}
	if strings.Join(res.Keywords, ",") != "VAT,tax" {
		t.Errorf("keywords not cleaned: %q", res.Keywords)
	}

	out := validOutput()
	out.CategoryCode, out.SubcategoryCode, out.RecordType = "UNKNOWN", nil, "other"
	if res, err := Validate(out, testTaxonomy(t)); err != nil || res.CategoryID != 3 || res.SubcategoryID != nil {
		t.Errorf("UNKNOWN without subcategory must be valid: %+v %v", res, err)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := map[string]func(o *dto.NormalizeOutput){
		"invented category":             func(o *dto.NormalizeOutput) { o.CategoryCode = "BUSINESS_TAX_POLICY" },
		"subcategory as category":       func(o *dto.NormalizeOutput) { o.CategoryCode = "FIN.TAX" },
		"subcategory of other category": func(o *dto.NormalizeOutput) { o.SubcategoryCode = ptr("HEALTH.MEDICINES") },
		"unknown subcategory":           func(o *dto.NormalizeOutput) { o.SubcategoryCode = ptr("FIN.CRYPTO") },
		"category as subcategory":       func(o *dto.NormalizeOutput) { o.SubcategoryCode = ptr("HEALTH") },
		"UNKNOWN with subcategory":      func(o *dto.NormalizeOutput) { o.CategoryCode = "UNKNOWN" },
		"record type not allowed":       func(o *dto.NormalizeOutput) { o.RecordType = "BLOG" },
		"confidence missing":            func(o *dto.NormalizeOutput) { o.Confidence = nil },
		"confidence above 1":            func(o *dto.NormalizeOutput) { o.Confidence = ptr(1.5) },
		"confidence below 0":            func(o *dto.NormalizeOutput) { o.Confidence = ptr(-0.1) },
		"published_at not a date":       func(o *dto.NormalizeOutput) { o.PublishedAt = ptr("last week") },
	}
	tax := testTaxonomy(t)
	for name, mutate := range cases {
		out := validOutput()
		mutate(&out)
		_, err := Validate(out, tax)
		var verr *model.ValidationError
		if !errors.As(err, &verr) {
			t.Errorf("%s: expected ValidationError, got %v", name, err)
		}
	}
}

func TestParseOutput(t *testing.T) {
	good := `{"record_type":"NEWS","category_code":"FIN","subcategory_code":null,"title":"T","language":"en",
		"published_at":null,"content":"C","summary":"S","keywords":["a"],"confidence":0.5}`
	for name, raw := range map[string]string{
		"plain":  good,
		"fenced": "```json\n" + good + "\n```",
		"think":  "<think>The page is about {taxes}.</think>\n" + good,
		"prose":  "Here is the JSON:\n" + good + "\nDone.",
	} {
		out, err := ParseOutput(raw)
		if err != nil || out.CategoryCode != "FIN" || out.SubcategoryCode != nil || *out.Confidence != 0.5 {
			t.Errorf("%s: out=%+v err=%v", name, out, err)
		}
	}
	for _, raw := range []string{"", "no json here", `{"record_type": "NEWS"`, `{"confidence": "high"}`} {
		if _, err := ParseOutput(raw); err == nil {
			t.Errorf("ParseOutput(%q) should fail", raw)
		}
	}
}

func TestOutputSchemaUsesTaxonomyEnums(t *testing.T) {
	schema := OutputSchema(testTaxonomy(t))
	raw, _ := json.Marshal(schema)
	for _, want := range []string{`"FIN.TAX"`, `"UNKNOWN"`, `"REGULATION"`, `"additionalProperties":false`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("schema is missing %s", want)
		}
	}
}

// fakeServer answers /chat/completions with the given replies in order and
// records each request body.
func fakeServer(t *testing.T, replies ...string) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var bodies []map[string]any
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Method != http.MethodPost {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		bodies = append(bodies, body)
		reply := replies[min(int(n.Add(1))-1, len(replies)-1)]
		if strings.HasPrefix(reply, "status:") {
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"content": reply}, "finish_reason": "stop"}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &bodies
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestNewClientSelectsRuntime(t *testing.T) {
	for runtime, wantFormat := range map[string]string{RuntimeVLLM: "json_schema", RuntimeLlamaCpp: "json_object"} {
		srv, bodies := fakeServer(t, mustJSON(validOutput()))
		client, err := NewClient(runtime, ClientConfig{BaseURL: srv.URL + "/v1", Model: "qwen-test", Timeout: 5 * time.Second, MaxTokens: 512})
		if err != nil {
			t.Fatal(err)
		}
		if client.Runtime() != runtime || client.Model() != "qwen-test" {
			t.Errorf("%s: Runtime/Model = %s/%s", runtime, client.Runtime(), client.Model())
		}
		out, err := client.Normalize(context.Background(), dto.NormalizeInput{SourceURL: "https://example.gov/a", Content: "text"},
			OutputSchema(testTaxonomy(t)))
		if err != nil || out.CategoryCode != "FIN" {
			t.Fatalf("%s: out=%+v err=%v", runtime, out, err)
		}
		body := (*bodies)[0]
		format := body["response_format"].(map[string]any)
		if format["type"] != wantFormat || body["model"] != "qwen-test" || body["temperature"] != 0.0 {
			t.Errorf("%s: unexpected request %v", runtime, body)
		}
		msgs := body["messages"].([]any)
		if user := msgs[1].(map[string]any)["content"].(string); !strings.Contains(user, `"source_url":"https://example.gov/a"`) {
			t.Errorf("%s: user message must carry the input JSON, got %s", runtime, user)
		}
	}
	if _, err := NewClient("openai", ClientConfig{}); err == nil {
		t.Fatal("unknown runtime must be rejected")
	}
}

func TestNormalizerRetriesAndValidates(t *testing.T) {
	invalid := validOutput()
	invalid.CategoryCode = "BUSINESS_TAX_POLICY"
	srv, bodies := fakeServer(t, "status:503", "not json", mustJSON(invalid), mustJSON(validOutput()))
	client, _ := NewClient(RuntimeVLLM, ClientConfig{BaseURL: srv.URL + "/v1", Timeout: 5 * time.Second})
	n := NewNormalizer(client, 1, 3, 10, slog.New(slog.NewTextHandler(io.Discard, nil)))
	n.retryDelay = time.Millisecond

	res, err := n.Normalize(context.Background(), dto.NormalizeInput{Content: "0123456789abcdef"}, testTaxonomy(t))
	if err != nil {
		t.Fatal(err)
	}
	if res.CategoryID != 1 || len(*bodies) != 4 {
		t.Fatalf("expected success on the 4th call: res=%+v calls=%d", res, len(*bodies))
	}
	var in dto.NormalizeInput
	json.Unmarshal([]byte((*bodies)[0]["messages"].([]any)[1].(map[string]any)["content"].(string)), &in)
	if in.Content != "0123456789" || len(in.Categories) != 3 || len(in.RecordTypes) != 3 {
		t.Errorf("input must be truncated and carry the taxonomy: %+v", in)
	}

	srv, _ = fakeServer(t, mustJSON(invalid))
	client, _ = NewClient(RuntimeLlamaCpp, ClientConfig{BaseURL: srv.URL + "/v1", Timeout: 5 * time.Second})
	n = NewNormalizer(client, 1, 1, 1000, slog.New(slog.NewTextHandler(io.Discard, nil)))
	n.retryDelay = time.Millisecond
	if _, err := n.Normalize(context.Background(), dto.NormalizeInput{}, testTaxonomy(t)); err == nil {
		t.Fatal("invalid taxonomy output must never be returned as a result")
	}
}
