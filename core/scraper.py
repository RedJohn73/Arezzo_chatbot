# core/scraper.py
# Incremental Drupal Crawler for Comune di Arezzo
# Full updated version with max_pages, max_depth, async, deduplication
# Compatible with Streamlit Cloud

import asyncio
import aiohttp
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import os

BASE_URL = "https://www.comune.arezzo.it"
DOMAIN = urlparse(BASE_URL).netloc

CRAWL_STATE_PATH = "data/crawl_state.json"     # url -> checksum
CRAWLED_DOCS_PATH = "data/comune_arezzo_dump.json"

MAX_CONCURRENCY = 8


# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------

def normalize_url(url: str) -> str:
    """Normalize URLs for deduplication."""
    p = urlparse(url)
    p = p._replace(fragment="")
    path = p.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return p._replace(path=path).geturl()


def is_valid_url(url: str) -> bool:
    """Filter out external links and useless file types."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc and parsed.netloc != DOMAIN:
        return False
    if any(url.lower().endswith(ext) for ext in [".jpg",".jpeg",".png",".gif",".pdf",".zip",".doc",".docx"]):
        return False
    return True


async def fetch(session: aiohttp.ClientSession, url: str):
    """Async HTML fetch."""
    try:
        async with session.get(url, timeout=15) as resp:
            if resp.status != 200:
                return None
            return await resp.text()
    except:
        return None


def extract_page(html: str, url: str):
    """Extracts text + metadata + breadcrumbs."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove layout clutter
    for tag in soup(["script","style","header","footer","nav"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ", strip=True).split())
    if len(text) < 100:
        return None

    title = soup.title.string.strip() if soup.title and soup.title.string else "Senza titolo"

    md = soup.find("meta", attrs={"name":"description"})
    meta_desc = md["content"].strip() if md and md.get("content") else ""

    mk = soup.find("meta", attrs={"name":"keywords"})
    meta_keywords = mk["content"].strip() if mk and mk.get("content") else ""

    # Breadcrumbs
    crumbs = []
    cont = soup.select_one("nav.breadcrumb, ul.breadcrumb, ol.breadcrumb")
    if cont:
        for li in cont.find_all(["li","span","a"]):
            t = li.get_text(strip=True)
            if t:
                crumbs.append(t)

    # Content type inference
    path = urlparse(url).path.lower()
    b = [c.lower() for c in crumbs]

    if "notizie" in path or "news" in path or any("notizie" in c or "news" in c for c in b):
        ctype = "news"
    elif "bandi" in path or "gare" in path or any("bando" in c or "bandi" in c for c in b):
        ctype = "bando"
    elif "ordinanze" in path or any("ordinanze" in c for c in b):
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
# ASYNC INCREMENTAL CRAWLER
# ------------------------------------------------------

async def _crawl_incremental_async(max_pages: int, max_depth: int):
    """Core async crawler supporting incremental updates."""

    # Load crawl state (URL → checksum)
    if os.path.exists(CRAWL_STATE_PATH):
        with open(CRAWL_STATE_PATH, "r", encoding="utf-8") as f:
            crawl_state = json.load(f)
    else:
        crawl_state = {}

    # Load previous documents
    if os.path.exists(CRAWLED_DOCS_PATH):
        with open(CRAWLED_DOCS_PATH, "r", encoding="utf-8") as f:
            old_docs = json.load(f)
    else:
        old_docs = []

    queue = [(normalize_url(BASE_URL), 0)]
    visited = set()

    new_docs = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async with aiohttp.ClientSession() as session:

        while queue and len(visited) < max_pages:

            batch = []
            while queue and len(batch) < MAX_CONCURRENCY:
                batch.append(queue.pop(0))

            tasks = []
            for url, depth in batch:
                if url in visited or depth > max_depth:
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

                visited.add(url)

                checksum = md5(html)

                # Skip if unchanged
                if url in crawl_state and crawl_state[url] == checksum:
                    continue

                page = extract_page(html, url)
                if page:
                    new_docs.append(page)

                # Update checksum
                crawl_state[url] = checksum

                # BFS expansion
                if depth < max_depth:
                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.find_all("a", href=True):
                        link = normalize_url(urljoin(url, a["href"]))
                        if is_valid_url(link) and link not in visited:
                            queue.append((link, depth + 1))

    # Merge: overwrite updated pages
    final_docs = {doc["url"]: doc for doc in old_docs}
    for d in new_docs:
        final_docs[d["url"]] = d

    final_list = list(final_docs.values())

    # Save docs
    os.makedirs("data", exist_ok=True)
    with open(CRAWLED_DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    # Save crawl state
    with open(CRAWL_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(crawl_state, f, ensure_ascii=False, indent=2)

    return len(new_docs)


# ------------------------------------------------------
# STREAMLIT WRAPPER
# ------------------------------------------------------

def incremental_crawl(max_pages=400, max_depth=4):
    """
    Streamlit-safe wrapper.
    Accepts dynamic crawling params.
    """
    return asyncio.run(_crawl_incremental_async(max_pages, max_depth))
