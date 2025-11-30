# core/scraper.py
# Incremental Drupal Crawler for Comune di Arezzo
# Features:
#   - Multi-level BFS crawling
#   - Async parallel HTTP fetch
#   - Incremental crawling (only new or modified pages)
#   - MD5 checksums for change detection
#   - URL normalization
#   - Breadcrumb extraction
#   - Content-type classification (news, bando, ordinanza, pagina)
#   - Safe for Streamlit Cloud

import asyncio
import aiohttp
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import os

BASE_URL = "https://www.comune.arezzo.it"
DOMAIN = urlparse(BASE_URL).netloc

CRAWL_STATE_PATH = "data/crawl_state.json"  # url → checksum
CRAWLED_DOCS_PATH = "data/comune_arezzo_dump.json"

MAX_PAGES = 600         # max total pages visited
MAX_DEPTH = 4           # BFS depth
MAX_CONCURRENCY = 8     # parallel HTTP requests


# ------------------------------------------------------
# Helper utilities
# ------------------------------------------------------

def normalize_url(url: str) -> str:
    """Normalize URLs for consistent hashing and duplicate removal."""
    p = urlparse(url)
    p = p._replace(fragment="")  # remove #anchor
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    p = p._replace(path=path)
    return p.geturl()


def is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc and parsed.netloc != DOMAIN:
        return False
    if any(url.lower().endswith(ext) for ext in [".jpg",".jpeg",".png",".pdf",".gif",".zip",".doc",".docx"]):
        return False
    return True


async def fetch(session, url):
    """Fetch HTML with aiohttp."""
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            return await resp.text()
    except:
        return None


def extract_page(html, url):
    """Extract text, title, metadata, breadcrumbs."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content
    for tag in soup(["script","style","header","footer","nav"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    if not text:
        return None

    title = soup.title.string.strip() if soup.title and soup.title.string else "Senza titolo"

    md = soup.find("meta", attrs={"name":"description"})
    meta_desc = md["content"].strip() if md and md.get("content") else ""

    mk = soup.find("meta", attrs={"name":"keywords"})
    meta_keywords = mk["content"].strip() if mk and mk.get("content") else ""

    # Breadcrumbs extraction (Drupal compatible)
    crumbs = []
    cont = soup.select_one("nav.breadcrumb, ul.breadcrumb, ol.breadcrumb")
    if cont:
        for li in cont.find_all(["li","a","span"]):
            t = li.get_text(strip=True)
            if t:
                crumbs.append(t)

    # Content type guess
    path = urlparse(url).path.lower()
    crumbs_l = [c.lower() for c in crumbs]

    if "notizie" in path or "news" in path or any("notizie" in c or "news" in c for c in crumbs_l):
        ctype = "news"
    elif "bandi" in path or "gare" in path or any("bando" in c or "bandi" in c for c in crumbs_l):
        ctype = "bando"
    elif "ordinanze" in path or any("ordinanze" in c for c in crumbs_l):
        ctype = "ordinanza"
    else:
        ctype = "pagina"

    return {
        "url": url,
        "title": title,
        "text": text,
        "meta_description": meta_desc,
        "meta_keywords": meta_keywords,
        "breadcrumbs": crumbs,
        "content_type": ctype
    }


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------
# Incremental Crawler
# ------------------------------------------------------

async def _crawl_incremental_async():
    """Incremental async BFS crawler."""
    # LOAD previous state (url → checksum)
    if os.path.exists(CRAWL_STATE_PATH):
        with open(CRAWL_STATE_PATH, "r", encoding="utf-8") as f:
            crawl_state = json.load(f)
    else:
        crawl_state = {}

    # LOAD previously crawled docs
    if os.path.exists(CRAWLED_DOCS_PATH):
        with open(CRAWLED_DOCS_PATH, "r", encoding="utf-8") as f:
            old_docs = json.load(f)
    else:
        old_docs = []

    # BFS queue
    queue = [(normalize_url(BASE_URL), 0)]
    visited = set()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    new_docs = []  # ONLY docs that changed

    async with aiohttp.ClientSession() as session:
        while queue and len(visited) < MAX_PAGES:
            batch = []
            # Take small parallelizable batch
            while queue and len(batch) < MAX_CONCURRENCY:
                batch.append(queue.pop(0))

            tasks = []
            for url, depth in batch:
                if url in visited or depth > MAX_DEPTH:
                    continue

                async def worker(u=url, d=depth):
                    async with semaphore:
                        html = await fetch(session, u)
                    return u, d, html

                tasks.append(worker())

            # Wait for batch results
            for coro in asyncio.as_completed(tasks):
                url, depth, html = await coro
                if not html:
                    continue

                visited.add(url)

                # Compute checksum
                checksum = md5(html)

                # Skip if unchanged
                if url in crawl_state and crawl_state[url] == checksum:
                    continue  # unchanged

                # Page changed or new → extract
                page = extract_page(html, url)
                if page:
                    new_docs.append(page)

                # Update checksum
                crawl_state[url] = checksum

                # Enqueue new links
                if depth < MAX_DEPTH:
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        link = normalize_url(urljoin(url, a["href"]))
                        if is_valid_url(link) and link not in visited:
                            queue.append((link, depth + 1))

    # Merge old docs + new/updated ones
    # Replace only updated pages (same URL)
    final_docs = {doc["url"]: doc for doc in old_docs}

    for d in new_docs:
        final_docs[d["url"]] = d  # overwrite if modified

    final_doc_list = list(final_docs.values())

    # SAVE docs and crawl-state
    os.makedirs("data", exist_ok=True)

    with open(CRAWLED_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_doc_list, f, ensure_ascii=False, indent=2)

    with open(CRAWL_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(crawl_state, f, ensure_ascii=False, indent=2)

    return len(new_docs)


def incremental_crawl():
    """
    Streamlit-safe wrapper.
    Returns the number of updated documents.
    """
    return asyncio.run(_crawl_incremental_async())

