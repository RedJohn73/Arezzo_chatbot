# core/scraper.py
# Advanced async crawler for Comune di Arezzo (Drupal)
# Supports:
#  - Multi-level crawling
#  - Automatic dedup (hash)
#  - Metadata extraction
#  - Breadcrumb extraction
#  - Content-type classification (news, bando, ordinanza, pagina)
#  - Async parallel requests
#  - Streamlit-safe wrapper

import asyncio
import hashlib
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

BASE_URL = "https://www.comune.arezzo.it"
DOMAIN = urlparse(BASE_URL).netloc

# SAFETY LIMITS FOR STREAMLIT CLOUD
MAX_PAGES_DEFAULT = 400
MAX_DEPTH_DEFAULT = 3
MAX_CONCURRENCY = 5   # parallel HTTP requests


# --------------------------------------------------------
# URL VALIDATION
# --------------------------------------------------------
def is_valid_url(url: str) -> bool:
    """Accept only internal HTTP(S) pages, skip static assets."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc and parsed.netloc != DOMAIN:
        return False
    # Skip static files
    if any(url.lower().endswith(ext) for ext in (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx")):
        return False
    return True


# --------------------------------------------------------
# TEXT + METADATA EXTRACTION
# --------------------------------------------------------
def extract_text_and_meta(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # Remove irrelevant elements
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    # Clean text
    text = " ".join(soup.get_text(separator=" ", strip=True).split())

    # Title
    title = soup.title.string.strip() if soup.title and soup.title.string else "Senza titolo"

    # Meta description
    meta_desc = None
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        meta_desc = desc_tag["content"].strip()

    # Keywords
    meta_kw = None
    kw_tag = soup.find("meta", attrs={"name": "keywords"})
    if kw_tag and kw_tag.get("content"):
        meta_kw = kw_tag["content"].strip()

    # Breadcrumbs (Drupal)
    breadcrumbs = []
    crumb_container = soup.select_one("nav.breadcrumb, ol.breadcrumb, ul.breadcrumb")
    if crumb_container:
        for li in crumb_container.find_all(["li", "a", "span"]):
            txt = li.get_text(strip=True)
            if txt:
                breadcrumbs.append(txt)

    return text, title, meta_desc, meta_kw, breadcrumbs


# --------------------------------------------------------
# CONTENT-TYPE CLASSIFICATION
# --------------------------------------------------------
def guess_content_type(url: str, breadcrumbs):
    path = urlparse(url).path.lower()

    if any(seg in path for seg in ["notizie", "news"]):
        return "news"
    if any(seg in path for seg in ["bandi", "gare"]):
        return "bando"
    if "ordinanze" in path:
        return "ordinanza"

    # breadcrumbs check
    crumbs = [c.lower() for c in breadcrumbs]
    if any("notizie" in c or "news" in c for c in crumbs):
        return "news"
    if any("bando" in c or "bandi" in c or "gare" in c for c in crumbs):
        return "bando"
    if any("ordinanze" in c for c in crumbs):
        return "ordinanza"

    return "pagina"


# --------------------------------------------------------
# URL NORMALISATION
# --------------------------------------------------------
def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    parsed = parsed._replace(path=path)
    return parsed.geturl()


# --------------------------------------------------------
# ASYNC FETCH
# --------------------------------------------------------
async def fetch(session: aiohttp.ClientSession, url: str):
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            return await resp.text()
    except:
        return None


# --------------------------------------------------------
# CORE ASYNC CRAWLER
# --------------------------------------------------------
async def _crawl_async(start_url: str, max_pages: int, max_depth: int, allowed_types):
    start_url = normalize_url(start_url)

    visited_urls = set()
    seen_hashes = set()
    results = []

    # BFS queue: (url, depth)
    queue = [(start_url, 0)]

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:
        while queue and len(visited_urls) < max_pages:

            # Take a small batch
            batch = []
            while queue and len(batch) < MAX_CONCURRENCY:
                batch.append(queue.pop(0))

            tasks = []

            for url, depth in batch:
                if url in visited_urls or depth > max_depth:
                    continue

                async def worker(u=url, d=depth):
                    async with semaphore:
                        html = await fetch(session, u)
                    return u, d, html

                tasks.append(worker())

            for coro in asyncio.as_completed(tasks):
                url, depth, html = await coro
                if not html:
                    continue

                visited_urls.add(url)

                text, title, meta_desc, meta_kw, breadcrumbs = extract_text_and_meta(html)

                if not text or len(text) < 200:  # skip thin pages
                    continue

                # dedup by hash of text
                content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                content_type = guess_content_type(url, breadcrumbs)

                include = True
                if allowed_types and content_type not in allowed_types:
                    include = False

                if include:
                    results.append({
                        "url": url,
                        "title": title,
                        "text": text,
                        "meta_description": meta_desc,
                        "meta_keywords": meta_kw,
                        "breadcrumbs": breadcrumbs,
                        "content_type": content_type,
                    })

                # Crawl deeper
                if depth < max_depth and len(visited_urls) < max_pages:
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        link = normalize_url(urljoin(url, a["href"]))
                        if is_valid_url(link) and link not in visited_urls:
                            queue.append((link, depth + 1))

    return results


# --------------------------------------------------------
# STREAMLIT-SAFE WRAPPER
# --------------------------------------------------------
def crawl_comune_arezzo(max_pages=MAX_PAGES_DEFAULT, max_depth=MAX_DEPTH_DEFAULT, content_types=None):
    """
    Wrapper sincrono per Streamlit.
    content_types: ["news", "bando", "ordinanza", "pagina"]
    """
    allowed_types = set(content_types) if content_types else None
    return asyncio.run(
        _crawl_async(BASE_URL, max_pages, max_depth, allowed_types)
    )
