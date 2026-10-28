#!/usr/bin/env python3
"""Export ggkip records from the HTTP API to one JSON file.

Each record gets readable names next to its ids (category and subcategory
codes, organization, site domain) and, unless --no-content is given, the full
content and metadata from GET /v1/records/{id}. Standard library only.

    python3 scripts/export_records.py --out records.json
    python3 scripts/export_records.py --site-id 8 --out stat_uz.json
    python3 scripts/export_records.py --country-id 242 --language uz --no-content
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

FILTERS = ["country_id", "organization_id", "site_id", "category_id", "subcategory_id",
           "record_type", "language", "published_from", "published_to"]


def get(base, path, params=None):
    url = base.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def get_all(base, path, params=None):
    """Follows next_cursor until the list is exhausted."""
    params = dict(params or {}, limit=200)
    while True:
        page = get(base, path, params)
        yield from page["items"]
        if not page.get("next_cursor"):
            return
        params["cursor"] = page["next_cursor"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8090", help="ggkip base URL")
    ap.add_argument("--out", default="records.json", help="output file ('-' for stdout)")
    ap.add_argument("--no-content", action="store_true", help="skip per-record content (one request less per record)")
    for f in FILTERS:
        ap.add_argument("--" + f.replace("_", "-"), dest=f)
    args = ap.parse_args()
    filters = {f: getattr(args, f) for f in FILTERS if getattr(args, f)}

    categories = {c["id"]: c for c in get_all(args.api, "/v1/categories")}
    organizations = {o["id"]: o for o in get_all(args.api, "/v1/organizations")}
    sites = {s["id"]: s for s in get_all(args.api, "/v1/sites")}

    out = []
    for r in get_all(args.api, "/v1/records", filters):
        if not args.no_content:
            r = get(args.api, f"/v1/records/{r['id']}")
        cat = categories.get(r["category_id"], {})
        sub = categories.get(r["subcategory_id"], {}) if r.get("subcategory_id") else {}
        r["category_code"], r["category_name"] = cat.get("code"), cat.get("name")
        r["subcategory_code"], r["subcategory_name"] = sub.get("code"), sub.get("name")
        r["organization_name"] = organizations.get(r["organization_id"], {}).get("name")
        r["site_domain"] = sites.get(r["site_id"], {}).get("domain")
        out.append(r)
        if len(out) % 100 == 0:
            print(f"{len(out)} records...", file=sys.stderr)

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {len(out)} records to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
