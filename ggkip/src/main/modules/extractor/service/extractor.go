package service

import (
	"bytes"
	"encoding/json"
	"net/url"
	"strings"
	"unicode"

	"golang.org/x/net/html"
	"golang.org/x/net/html/atom"

	"ggkip/src/main/modules/extractor/model"
)

// minMainTextChars: if <main>/<article> holds less text than this, the whole
// <body> is used instead (some sites wrap only a heading in <main>).
const minMainTextChars = 200

// skipTags never contain page content.
var skipTags = map[atom.Atom]bool{
	atom.Script: true, atom.Style: true, atom.Noscript: true, atom.Template: true, atom.Iframe: true,
	atom.Svg: true, atom.Canvas: true, atom.Nav: true, atom.Header: true, atom.Footer: true,
	atom.Aside: true, atom.Form: true, atom.Button: true, atom.Select: true, atom.Option: true,
	atom.Head: true, atom.Menu: true, atom.Dialog: true, atom.Object: true, atom.Embed: true,
}

var skipRoles = map[string]bool{
	"navigation": true, "banner": true, "contentinfo": true, "search": true, "menu": true,
	"menubar": true, "dialog": true, "alertdialog": true, "complementary": true,
}

// skipClassTokens drop layout blocks by class or id (cookie banners, menus...).
var skipClassTokens = []string{
	"cookie", "consent", "gdpr", "navbar", "menu", "breadcrumb", "sidebar", "footer",
	"site-header", "skip-link", "share", "social", "pagination", "banner",
}

var blockTags = map[atom.Atom]bool{
	atom.P: true, atom.Div: true, atom.Section: true, atom.Article: true, atom.Main: true, atom.Li: true,
	atom.Ul: true, atom.Ol: true, atom.Dl: true, atom.Dt: true, atom.Dd: true, atom.Tr: true,
	atom.Table: true, atom.Blockquote: true, atom.Pre: true, atom.Figcaption: true, atom.Address: true,
	atom.H1: true, atom.H2: true, atom.H3: true, atom.H4: true, atom.H5: true, atom.H6: true,
	atom.Br: true, atom.Hr: true, atom.Caption: true, atom.Summary: true, atom.Details: true,
}

var headingTags = map[atom.Atom]bool{
	atom.H1: true, atom.H2: true, atom.H3: true, atom.H4: true, atom.H5: true, atom.H6: true,
}

// Extract converts a UTF-8 body into clean text. pageURL is used to resolve <base href>.
func Extract(body []byte, contentType string, pageURL *url.URL) model.Content {
	switch {
	case contentType == "text/plain":
		return model.Content{Text: normalizeText(string(body))}
	case contentType == "application/json" || strings.HasSuffix(contentType, "+json"):
		return extractJSON(body)
	default:
		return extractHTML(body, pageURL)
	}
}

func extractHTML(body []byte, pageURL *url.URL) model.Content {
	doc, err := html.Parse(bytes.NewReader(body))
	if err != nil {
		return model.Content{}
	}
	var c model.Content
	var mainNode, articleNode, bodyNode *html.Node
	base := pageURL

	var walk func(n *html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch n.DataAtom {
			case atom.Html:
				c.Language = normalizeLang(attr(n, "lang"))
			case atom.Title:
				if c.Title == "" {
					c.Title = collapseSpaces(textOf(n))
				}
			case atom.Base:
				if href := attr(n, "href"); href != "" && base != nil {
					if b, err := base.Parse(href); err == nil {
						base = b
					}
				}
			case atom.Meta:
				if strings.EqualFold(attr(n, "name"), "robots") {
					v := strings.ToLower(attr(n, "content"))
					c.NoIndex = c.NoIndex || strings.Contains(v, "noindex") || strings.Contains(v, "none")
					c.NoFollow = c.NoFollow || strings.Contains(v, "nofollow") || strings.Contains(v, "none")
				}
			case atom.A:
				if href := strings.TrimSpace(attr(n, "href")); href != "" && !strings.HasPrefix(href, "#") {
					c.Links = append(c.Links, href)
				}
			case atom.Main:
				if mainNode == nil {
					mainNode = n
				}
			case atom.Article:
				if articleNode == nil {
					articleNode = n
				}
			case atom.Body:
				bodyNode = n
			}
		}
		for ch := n.FirstChild; ch != nil; ch = ch.NextSibling {
			walk(ch)
		}
	}
	walk(doc)

	// Links are resolved here so <base href> is respected; the crawler normalizes them.
	if base != nil {
		for i, l := range c.Links {
			if u, err := base.Parse(l); err == nil {
				c.Links[i] = u.String()
			}
		}
	}

	root := bodyNode
	for _, candidate := range []*html.Node{mainNode, articleNode} {
		if candidate != nil {
			if text, _ := renderText(candidate); len([]rune(text)) >= minMainTextChars {
				root = candidate
				break
			}
		}
	}
	if root != nil {
		c.Text, c.Headings = renderText(root)
	}
	if c.Title == "" && len(c.Headings) > 0 {
		c.Title = c.Headings[0]
	}
	return c
}

// renderText walks n and returns its visible text with one block per line,
// plus the headings found inside it.
func renderText(n *html.Node) (string, []string) {
	var b strings.Builder
	var headings []string
	seenHeading := map[string]bool{}

	newline := func() {
		if b.Len() > 0 && !strings.HasSuffix(b.String(), "\n") {
			b.WriteByte('\n')
		}
	}
	var walk func(n *html.Node, inPre bool)
	walk = func(n *html.Node, inPre bool) {
		switch n.Type {
		case html.TextNode:
			if inPre {
				b.WriteString(n.Data)
			} else {
				// Outside <pre>, source line breaks are just spaces, as in a browser.
				b.WriteString(strings.Map(func(r rune) rune {
					if unicode.IsSpace(r) {
						return ' '
					}
					return r
				}, n.Data))
			}
			return
		case html.ElementNode:
			if skipElement(n) {
				return
			}
			if headingTags[n.DataAtom] {
				if h := collapseSpaces(textOf(n)); h != "" && !seenHeading[h] {
					seenHeading[h] = true
					headings = append(headings, h)
				}
			}
			if blockTags[n.DataAtom] {
				newline()
			}
			if n.DataAtom == atom.Li {
				b.WriteString("- ")
			}
			inPre = inPre || n.DataAtom == atom.Pre || n.DataAtom == atom.Textarea
		}
		for ch := n.FirstChild; ch != nil; ch = ch.NextSibling {
			walk(ch, inPre)
		}
		if n.Type == html.ElementNode {
			switch {
			case blockTags[n.DataAtom]:
				newline()
			case n.DataAtom == atom.Td || n.DataAtom == atom.Th:
				b.WriteString(" | ")
			}
		}
	}
	walk(n, false)
	return normalizeText(b.String()), headings
}

func skipElement(n *html.Node) bool {
	if skipTags[n.DataAtom] {
		return true
	}
	if _, hidden := attrOK(n, "hidden"); hidden {
		return true
	}
	if strings.EqualFold(attr(n, "aria-hidden"), "true") || skipRoles[strings.ToLower(attr(n, "role"))] {
		return true
	}
	style := strings.ReplaceAll(strings.ToLower(attr(n, "style")), " ", "")
	if strings.Contains(style, "display:none") || strings.Contains(style, "visibility:hidden") {
		return true
	}
	names := strings.ToLower(attr(n, "class") + " " + attr(n, "id"))
	for token := range strings.FieldsSeq(names) {
		for _, bad := range skipClassTokens {
			if strings.Contains(token, bad) {
				return true
			}
		}
	}
	return false
}

func attr(n *html.Node, key string) string {
	v, _ := attrOK(n, key)
	return v
}

func attrOK(n *html.Node, key string) (string, bool) {
	for _, a := range n.Attr {
		if a.Namespace == "" && strings.EqualFold(a.Key, key) {
			return a.Val, true
		}
	}
	return "", false
}

func textOf(n *html.Node) string {
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			b.WriteString(n.Data)
		}
		if n.Type == html.ElementNode && (n.DataAtom == atom.Script || n.DataAtom == atom.Style) {
			return
		}
		for ch := n.FirstChild; ch != nil; ch = ch.NextSibling {
			walk(ch)
		}
	}
	walk(n)
	return b.String()
}

// extractJSON keeps string values that look like prose (3+ words or CJK
// text), which is how JSON APIs on government sites usually carry content.
func extractJSON(body []byte) model.Content {
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		return model.Content{}
	}
	var c model.Content
	var lines []string
	var walk func(key string, v any)
	walk = func(key string, v any) {
		switch t := v.(type) {
		case map[string]any:
			for k, val := range t {
				walk(k, val)
			}
		case []any:
			for _, val := range t {
				walk(key, val)
			}
		case string:
			s := collapseSpaces(t)
			if c.Title == "" && (key == "title" || key == "name") && s != "" {
				c.Title = s
			}
			if WordCount(s) >= 3 {
				lines = append(lines, s)
			}
		}
	}
	walk("", v)
	c.Text = normalizeText(strings.Join(lines, "\n"))
	return c
}

// normalizeText collapses whitespace inside lines, drops empty lines and
// repeated consecutive lines.
func normalizeText(s string) string {
	var out []string
	for line := range strings.SplitSeq(s, "\n") {
		line = strings.Trim(collapseSpaces(line), " |")
		if line == "" || line == "-" || (len(out) > 0 && out[len(out)-1] == line) {
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

func collapseSpaces(s string) string {
	return strings.Join(strings.FieldsFunc(s, unicode.IsSpace), " ")
}

func normalizeLang(v string) string {
	v = strings.ToLower(strings.TrimSpace(v))
	if len(v) > 10 {
		v = v[:10]
	}
	return v
}
