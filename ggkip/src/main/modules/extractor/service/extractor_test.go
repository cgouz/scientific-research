package service

import (
	"net/url"
	"strings"
	"testing"

	"ggkip/src/main/modules/extractor/model"
)

const page = `<!doctype html>
<html lang="en-GB">
<head>
  <title>  VAT changes  </title>
  <meta name="robots" content="index, follow">
  <style>.x { color: red }</style>
  <script>var tracking = "do not extract";</script>
</head>
<body>
  <header class="site-header"><a href="/">Home</a> Header text</header>
  <nav><ul><li><a href="/news">News</a></li><li><a href="/about">About</a></li></ul></nav>
  <div class="cookie-banner">We use cookies. Accept?</div>
  <main>
    <h1>Changes to VAT rates</h1>
    <p>From   1 January, VAT rates for certain businesses will change.
       The standard rate stays at 20 percent.</p>
    <h2>Who is affected</h2>
    <ul><li>Small businesses</li><li>Exporters</li></ul>
    <table><tr><th>Rate</th><th>Value</th></tr><tr><td>Standard</td><td>20%</td></tr></table>
    <p hidden>hidden text</p>
    <p style="display: none">also hidden</p>
    <p>Read the <a href="guide?utm_source=x">full guide</a> for details on registration,
       invoices and deadlines that apply to every business in the scheme.</p>
  </main>
  <aside>Related links</aside>
  <footer>Crown copyright</footer>
</body>
</html>`

func TestExtractHTML(t *testing.T) {
	base, _ := url.Parse("https://www.example.gov/tax/vat")
	c := Extract([]byte(page), "text/html", base)

	if c.Title != "VAT changes" {
		t.Errorf("Title = %q", c.Title)
	}
	if c.Language != "en-gb" {
		t.Errorf("Language = %q", c.Language)
	}
	if strings.Join(c.Headings, "|") != "Changes to VAT rates|Who is affected" {
		t.Errorf("Headings = %q", c.Headings)
	}
	for _, want := range []string{
		"Changes to VAT rates",
		"From 1 January, VAT rates for certain businesses will change. The standard rate stays at 20 percent.",
		"- Small businesses",
		"Rate | Value",
		"Standard | 20%",
	} {
		if !strings.Contains(c.Text, want) {
			t.Errorf("Text is missing %q:\n%s", want, c.Text)
		}
	}
	for _, unwanted := range []string{"tracking", "color", "Header text", "News", "cookies", "hidden", "Related", "Crown"} {
		if strings.Contains(c.Text, unwanted) {
			t.Errorf("Text must not contain %q:\n%s", unwanted, c.Text)
		}
	}
	links := strings.Join(c.Links, " ")
	for _, want := range []string{"https://www.example.gov/news", "https://www.example.gov/tax/guide?utm_source=x"} {
		if !strings.Contains(links, want) {
			t.Errorf("Links %v are missing %s (links come from the whole document)", c.Links, want)
		}
	}
	if c.NoIndex || c.NoFollow {
		t.Error("robots meta says index, follow")
	}
}

func TestExtractFallsBackToBodyAndHonoursBaseAndMeta(t *testing.T) {
	html := `<html><head><base href="https://cdn.example.gov/root/"><meta name="robots" content="noindex,nofollow"></head>
	<body><main><h1>Title only</h1></main><div><p>` + strings.Repeat("Body text sentence. ", 20) + `</p>
	<a href="page">p</a></div></body></html>`
	c := Extract([]byte(html), "text/html", mustURL("https://example.gov/x"))
	if !strings.Contains(c.Text, "Body text sentence.") {
		t.Errorf("small <main> must fall back to <body>: %q", c.Text)
	}
	if len(c.Links) != 1 || c.Links[0] != "https://cdn.example.gov/root/page" {
		t.Errorf("links must resolve against <base href>: %v", c.Links)
	}
	if !c.NoIndex || !c.NoFollow {
		t.Error("meta robots noindex,nofollow not detected")
	}
	if c.Title != "Title only" {
		t.Errorf("title falls back to the first heading, got %q", c.Title)
	}
}

func TestExtractPlainTextAndJSON(t *testing.T) {
	c := Extract([]byte("  line one  \n\n\nline   two\nline   two\n"), "text/plain", nil)
	if c.Text != "line one\nline two" {
		t.Errorf("plain text = %q", c.Text)
	}
	c = Extract([]byte(`{"title":"Notice","items":[{"body":"The ministry announced new rules today."},{"id":"x1"}]}`),
		"application/json", nil)
	if c.Title != "Notice" || c.Text != "The ministry announced new rules today." {
		t.Errorf("json content = %+v", c)
	}
}

func TestFilter(t *testing.T) {
	f := Filter{MinChars: 40, MinWords: 8}
	long := "This page explains how to register a business with the tax office."
	cases := []struct {
		c      model.Content
		ok     bool
		reason string
	}{
		{model.Content{Text: ""}, false, ReasonEmpty},
		{model.Content{Text: "   \n "}, false, ReasonEmpty},
		{model.Content{Text: "Too short."}, false, ReasonTooShort},
		{model.Content{Text: strings.Repeat("x", 50)}, false, ReasonFewWords},
		{model.Content{Text: long, NoIndex: true}, false, ReasonNoIndex},
		{model.Content{Text: long}, true, ""},
		// Japanese has no spaces: each kanji/kana counts as a word.
		{model.Content{Text: "消費税の税率は令和元年十月一日から十パーセントに引き上げられました。対象となる事業者"}, true, ""},
	}
	f.MinChars = 30
	for _, c := range cases {
		ok, reason := f.Check(c.c)
		if ok != c.ok || reason != c.reason {
			t.Errorf("Check(%q) = %v %q, want %v %q", c.c.Text, ok, reason, c.ok, c.reason)
		}
	}
}

func mustURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		panic(err)
	}
	return u
}
