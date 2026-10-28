package service

import (
	"errors"
	"net"
	"net/url"
	"path"
	"strings"

	"ggkip/src/main/infrastructure"
	"ggkip/src/main/modules/crawler/model"
)

var ErrUnsupportedScheme = errors.New("only http and https URLs are crawled")

// URLNormalizer turns discovered links into a canonical form, so the same
// page always gets the same url_hash.
type URLNormalizer struct {
	trackingParams   map[string]bool
	trackingPrefixes []string
	skipExtensions   map[string]bool
}

func NewURLNormalizer(s model.Settings) *URLNormalizer {
	n := &URLNormalizer{trackingParams: map[string]bool{}, skipExtensions: map[string]bool{}}
	for _, p := range s.TrackingParams {
		n.trackingParams[strings.ToLower(p)] = true
	}
	for _, p := range s.TrackingParamPrefixes {
		n.trackingPrefixes = append(n.trackingPrefixes, strings.ToLower(p))
	}
	for _, e := range s.SkipExtensions {
		n.skipExtensions[strings.ToLower(e)] = true
	}
	return n
}

// Normalize resolves raw against base (which may be nil for absolute URLs) and:
// lowercases scheme and host, drops default ports, user info and fragments,
// removes dot segments and tracking parameters, sorts the remaining query
// parameters and gives an empty path a single "/". Other trailing slashes are
// kept, because servers may treat /a and /a/ as different pages.
func (n *URLNormalizer) Normalize(raw string, base *url.URL) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return "", err
	}
	if base != nil {
		u = base.ResolveReference(u)
	}
	u.Scheme = strings.ToLower(u.Scheme)
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", ErrUnsupportedScheme
	}
	host := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	if host == "" {
		return "", errors.New("URL has no host")
	}
	port := u.Port()
	if (u.Scheme == "http" && port == "80") || (u.Scheme == "https" && port == "443") {
		port = ""
	}
	u.Host = host
	if port != "" {
		u.Host = net.JoinHostPort(host, port)
	}
	u.User = nil
	u.Fragment, u.RawFragment = "", ""
	u.Path = cleanPath(u.Path)
	u.RawPath = ""

	if u.RawQuery != "" {
		q := u.Query()
		for key := range q {
			if n.isTracking(key) {
				q.Del(key)
			}
		}
		u.RawQuery = q.Encode()
	}
	u.ForceQuery = false
	return u.String(), nil
}

func (n *URLNormalizer) isTracking(key string) bool {
	key = strings.ToLower(key)
	if n.trackingParams[key] {
		return true
	}
	for _, p := range n.trackingPrefixes {
		if strings.HasPrefix(key, p) {
			return true
		}
	}
	return false
}

// IsSkippable reports whether the URL points at a file type that is never crawled.
func (n *URLNormalizer) IsSkippable(u *url.URL) bool {
	ext := strings.ToLower(path.Ext(u.Path))
	return ext != "" && n.skipExtensions[ext]
}

// cleanPath removes "." and ".." segments and duplicate slashes while keeping
// a trailing slash.
func cleanPath(p string) string {
	if p == "" || p == "/" {
		return "/"
	}
	trailing := strings.HasSuffix(p, "/")
	p = path.Clean("/" + p)
	if trailing && p != "/" {
		p += "/"
	}
	return p
}

// HashURL is SHA-256 of the normalized URL (pages.url_hash).
func HashURL(normalized string) []byte {
	return infrastructure.SHA256(normalized)
}

// Scope keeps a crawl on one site: the domain itself and its www. host.
// Other subdomains are separate sites with their own rows.
type Scope struct {
	domain string
}

func NewScope(domain string) Scope {
	return Scope{domain: strings.TrimPrefix(strings.ToLower(strings.TrimSuffix(domain, ".")), "www.")}
}

func (s Scope) Allows(u *url.URL) bool {
	host := strings.TrimSuffix(strings.ToLower(u.Hostname()), ".")
	return host == s.domain || host == "www."+s.domain
}
