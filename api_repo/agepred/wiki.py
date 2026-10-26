"""Live Wikipedia / Wikimedia Commons client for the API.

The date-parsing block in the middle is generated from pipeline/step2_photo_dates.py so the
live API reads photo dates with exactly the same rules that were used to build the
evaluation data.
"""
import calendar
import datetime as dt
import html
import os
import re
import time
import unicodedata
from urllib.parse import parse_qs, unquote, urlparse

import requests

UA = {"User-Agent": "EchoooAgeAssignment/1.0 (take-home assignment API; candidate Muhammad Shahzad; "
                    + os.environ.get("WIKI_CONTACT", "see GitHub repo") + ")"}
MAX_IMAGE_BYTES = 15_000_000
MAX_FALLBACK_IMAGES = 5


class WikiError(Exception):
    pass


class BadURL(WikiError):
    pass


class PageNotFound(WikiError):
    pass


class UpstreamError(WikiError):
    pass


# ---------------------------------------------------------------- photo dates (generated block)
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
BAD = re.compile(r"\b(circa|ca|c|about|approx\w*|around|before|after|between|early|late|"
                 r"mid|or|to|and|unknown|undated)\b|[?~/\u2013\u2014]", re.I)


def parse_photo_date(raw):
    """Return (date_lo, date_hi, precision) or (None, None, reason)."""
    if not raw:
        return None, None, "no_photo_date"
    m = re.search(r'datetime="([^"]+)"', raw)
    s = m.group(1) if m else html.unescape(re.sub(r"<[^>]+>", " ", raw))
    s = re.sub(r"\s+", " ", s).strip().rstrip(".")
    # the text says the date is NOT the capture date (upload / scan / publication date)
    if re.search(r"\b(upload\w*|scan\w*|publi\w*|clipping|stamp\w*)\b", s, re.I):
        return None, None, "unparseable_date"
    if (BAD.search(s) or re.search(r"\d{4}s", s)
            or len(re.findall(r"\b\d{4}\b", s)) != 1):
        return None, None, "unparseable_date"

    y = mo = d = None
    if (m := re.match(r"^(\d{4})[-:](\d{2})[-:](\d{2})(?:[T ].*)?$", s)):
        y, mo, d = map(int, m.groups())
    elif (m := re.match(r"^(\d{4})-(\d{2})$", s)):
        y, mo = map(int, m.groups())
    elif (m := re.match(r"^(\d{4})$", s)):
        y = int(m.group(1))
    elif (m := re.match(r"^(\d{1,2}) ([A-Za-z]+),? (\d{4})(?: .*)?$", s)) and m.group(2).lower() in MONTHS:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))
    elif (m := re.match(r"^([A-Za-z]+) (\d{1,2}),? (\d{4})(?: .*)?$", s)) and m.group(1).lower() in MONTHS:
        mo, d, y = MONTHS[m.group(1).lower()], int(m.group(2)), int(m.group(3))
    elif (m := re.match(r"^([A-Za-z]+),? (\d{4})$", s)) and m.group(1).lower() in MONTHS:
        mo, y = MONTHS[m.group(1).lower()], int(m.group(2))
    else:
        return None, None, "unparseable_date"

    try:
        if d:
            lo = hi = dt.date(y, mo, d); prec = "day"
        elif mo:
            lo = dt.date(y, mo, 1)
            hi = dt.date(y, mo, calendar.monthrange(y, mo)[1]); prec = "month"
        else:
            lo, hi = dt.date(y, 1, 1), dt.date(y, 12, 31); prec = "year"
    except ValueError:
        return None, None, "unparseable_date"
    if y < 1900 or lo > dt.date.today():
        return None, None, "implausible_date"
    return lo, min(hi, dt.date.today()), prec


def clean_raw(raw):
    """Second-pass cleaning for strings like 'Taken on 6 April 2006' or
    '20 September 2025 (according to Exif data)'. Returns None if the date is
    explicitly not a capture date (upload / scan / publication date)."""
    s = html.unescape(re.sub(r"<[^>]+>", " ", str(raw)))
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\([^)]*(upload|scan|publi)", s, re.I):
        return None
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"^(taken|photographed|created)( on)?\s+", "", s, flags=re.I)
    s = re.sub(r",\s*(?=\d{1,2}:\d{2})", " ", s)
    return re.sub(r"\s+", " ", s).strip()




def format_photo_date(lo, hi, prec):
    """'2016-05-03' for day precision, '2016-05' for month, '2016' for year."""
    if prec == "day":
        return lo.isoformat()
    if prec == "month":
        return lo.strftime("%Y-%m")
    return str(lo.year)


# ---------------------------------------------------------------- URL + API helpers
def parse_wikipedia_url(url):
    """Return (host, title). Only https://<lang>.wikipedia.org/wiki/<Title> style links are accepted."""
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        raise BadURL("Please send a full Wikipedia page link, e.g. https://en.wikipedia.org/wiki/Tom_Hanks")
    host = u.netloc.lower().split(":")[0].replace(".m.", ".")
    if not re.fullmatch(r"[a-z][a-z\-]*\.wikipedia\.org", host):
        raise BadURL("Only wikipedia.org page links are supported")
    if u.path.startswith("/wiki/"):
        title = unquote(u.path[len("/wiki/"):])
    elif u.path.startswith("/w/index.php"):
        title = parse_qs(u.query).get("title", [""])[0]
    else:
        raise BadURL("The link does not look like a Wikipedia article (expected /wiki/<Title>)")
    title = title.replace("_", " ").strip()
    if not title:
        raise BadURL("The link has no page title")
    return host, title


def _get_json(url, params, retries=3, timeout=20):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(1)
    raise UpstreamError("Wikipedia/Wikimedia did not answer; please try again in a moment")


def get_page_info(host, title):
    d = _get_json(f"https://{host}/w/api.php", {
        "action": "query", "titles": title, "redirects": 1,
        "prop": "pageimages|images", "piprop": "name|original|thumbnail", "pithumbsize": 960,
        "imlimit": 100})
    pages = d.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        raise PageNotFound(f"Wikipedia page not found: {title}")
    p = pages[0]
    return {"title": p["title"], "lead_file": p.get("pageimage"),
            "lead_thumb": (p.get("thumbnail") or {}).get("source"),
            "lead_original": (p.get("original") or {}).get("source"),
            "images": [i["title"] for i in p.get("images", [])]}


def _meta_value(md, key):
    v = md.get(key) if isinstance(md, dict) else None
    return v.get("value") if isinstance(v, dict) else None


def commons_info(files):
    """Commons imageinfo for file names: 960px thumbnail url, original url, mime, raw date."""
    out = {}
    files = list(dict.fromkeys(files))[:50]
    if not files:
        return out
    titles = ["File:" + f.replace("_", " ") for f in files]
    d = _get_json("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "titles": "|".join(titles), "redirects": 1, "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 960,
        "iiextmetadatafilter": "DateTimeOriginal"})
    q = d.get("query", {})
    norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
    redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
    pages = {p["title"]: p for p in q.get("pages", [])}
    for f, t in zip(files, titles):
        t2 = norm.get(t, t)
        p = pages.get(redir.get(t2, t2))
        if not p or p.get("missing") or not p.get("imageinfo"):
            continue
        ii = p["imageinfo"][0]
        out[f] = {"thumb_url": ii.get("thumburl") or ii.get("url"), "image_url": ii.get("url"),
                  "file_page": ii.get("descriptionurl"), "mime": ii.get("mime"),
                  "raw_date": _meta_value(ii.get("extmetadata"), "DateTimeOriginal")}
    return out


# ---------------------------------------------------------------- candidate photos
BAD_WORDS = ("logo", "signature", "autograph", "flag", "map", "icon", "symbol", "coat", "seal", "poster",
             "cover", "album", "banner", "screenshot", "commons-", "wiki", "ambox", "pictogram", "stamp")


def _norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def name_matched_files(title, image_titles, exclude=None):
    """Other images on the page whose file name contains part of the person's name."""
    base = re.sub(r"\s*\(.*?\)\s*", " ", title)
    tokens = [t for t in _norm(base).split() if len(t) >= 3]
    ex = _norm(exclude.rsplit(".", 1)[0]) if exclude else None
    out = []
    for t in image_titles:
        fname = t.split(":", 1)[-1]
        low = fname.lower()
        if not low.endswith((".jpg", ".jpeg", ".png")) or any(w in low for w in BAD_WORDS):
            continue
        nf = _norm(fname.rsplit(".", 1)[0])
        if nf == ex:
            continue
        if tokens and any(tok in nf.split() for tok in tokens):
            out.append(fname)
    return out


def resolve_candidates(host, page):
    """Ordered photos to try: the page's lead image first, then name-matched page images."""
    files, meta = [], []
    if page.get("lead_file"):
        files.append(page["lead_file"])
        meta.append({"file": page["lead_file"], "source": "lead_image"})
    for f in name_matched_files(page["title"], page.get("images", []), page.get("lead_file"))[:MAX_FALLBACK_IMAGES]:
        files.append(f)
        meta.append({"file": f, "source": "page_image_name_match"})
    info = commons_info(files) if files else {}
    cands = []
    for m in meta:
        ci = info.get(m["file"])
        if ci and ci["mime"] in ("image/jpeg", "image/png"):
            date = None
            lo, hi, prec = parse_photo_date(ci["raw_date"])
            if lo is None and prec == "unparseable_date":
                cleaned = clean_raw(ci["raw_date"])
                if cleaned:
                    lo, hi, prec = parse_photo_date(cleaned)
            if lo is not None:
                date = format_photo_date(lo, hi, prec)
            cands.append({**m, "thumb_url": ci["thumb_url"], "image_url": ci["image_url"],
                          "file_page": ci["file_page"], "photo_date": date})
        elif m["source"] == "lead_image" and page.get("lead_thumb"):
            # lead image that is not on Commons (e.g. a local fair-use file): no date available
            cands.append({**m, "thumb_url": page["lead_thumb"],
                          "image_url": page.get("lead_original") or page["lead_thumb"],
                          "file_page": None, "photo_date": None})
    return cands


def download_image(url):
    host = urlparse(url).netloc.lower()
    if not host.endswith("wikimedia.org"):
        raise UpstreamError("unexpected image host")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and r.content:
                if len(r.content) > MAX_IMAGE_BYTES:
                    raise UpstreamError("image too large")
                return r.content
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
        except requests.RequestException:
            pass
        time.sleep(1)
    raise UpstreamError("could not download the image")
