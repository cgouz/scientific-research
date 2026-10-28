package service

import (
	"bufio"
	"bytes"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Robots holds the robots.txt rules that apply to our user agent (RFC 9309).
type Robots struct {
	rules      []robotsRule
	CrawlDelay time.Duration
	Sitemaps   []string
}

type robotsRule struct {
	allow   bool
	pattern string
	re      *regexp.Regexp
}

// AllowAll is used when a site has no robots.txt (any 4xx response).
func AllowAll() *Robots { return &Robots{} }

// ParseRobots picks the group for our user agent token (the part of the
// User-Agent before "/"), falling back to the "*" group.
func ParseRobots(body []byte, userAgent string) *Robots {
	token := strings.ToLower(strings.TrimSpace(strings.SplitN(userAgent, "/", 2)[0]))

	type group struct {
		agents []string
		rules  []robotsRule
		delay  time.Duration
	}
	var groups []*group
	var current *group
	lastWasAgent := false
	robots := &Robots{}

	sc := bufio.NewScanner(bytes.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = line[:i]
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(key))
		value = strings.TrimSpace(value)

		switch key {
		case "user-agent":
			if !lastWasAgent || current == nil {
				current = &group{}
				groups = append(groups, current)
			}
			current.agents = append(current.agents, strings.ToLower(value))
			lastWasAgent = true
			continue
		case "allow", "disallow":
			if current != nil && value != "" {
				current.rules = append(current.rules, robotsRule{allow: key == "allow", pattern: value})
			}
		case "crawl-delay":
			if current != nil {
				if secs, err := strconv.ParseFloat(value, 64); err == nil && secs > 0 {
					current.delay = time.Duration(secs * float64(time.Second))
				}
			}
		case "sitemap":
			robots.Sitemaps = append(robots.Sitemaps, value)
		}
		lastWasAgent = false
	}

	var chosen, star *group
	for _, g := range groups {
		for _, a := range g.agents {
			if a == "*" && star == nil {
				star = g
			} else if a != "*" && a == token && chosen == nil {
				chosen = g
			}
		}
	}
	if chosen == nil {
		chosen = star
	}
	if chosen != nil {
		robots.rules = chosen.rules
		robots.CrawlDelay = chosen.delay
		for i := range robots.rules {
			robots.rules[i].re = compileRobotsPattern(robots.rules[i].pattern)
		}
	}
	return robots
}

// Allowed applies the longest matching rule; on a tie, Allow wins.
func (r *Robots) Allowed(u *url.URL) bool {
	target := u.EscapedPath()
	if u.RawQuery != "" {
		target += "?" + u.RawQuery
	}
	bestLen, allowed := -1, true
	for _, rule := range r.rules {
		if !rule.re.MatchString(target) {
			continue
		}
		if l := len(rule.pattern); l > bestLen || (l == bestLen && rule.allow) {
			bestLen, allowed = l, rule.allow
		}
	}
	return allowed
}

// compileRobotsPattern supports the "*" wildcard and a trailing "$" anchor.
func compileRobotsPattern(p string) *regexp.Regexp {
	anchored := strings.HasSuffix(p, "$")
	p = strings.TrimSuffix(p, "$")
	expr := "^" + strings.ReplaceAll(regexp.QuoteMeta(p), `\*`, ".*")
	if anchored {
		expr += "$"
	}
	return regexp.MustCompile(expr)
}
