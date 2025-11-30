import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

BASE_URL = "https://www.comune.arezzo.it"

# Max pages to crawl (per sicurezza su Streamlit Cloud)
MAX_PAGES = 200  

# Allowed domain
DOMAIN = urlparse(BASE_URL).netloc

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and parsed.netloc == DOMAIN

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, menus, footers
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())

def crawl_comune_arezzo():
    visited = set()
    to_visit = [BASE_URL]
    results = []

    while to_visit and len(visited) < MAX_PAGES:
        url = to_visit.pop(0)

        if url in visited:
            continue

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
        except:
            continue

        visited.add(url)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        text = extract_text(html)
        title = soup.title.string if soup.title else "Senza titolo"

        results.append({
            "url": url,
            "title": title,
            "text": text
        })

        # Trova tutti i link interni
        for link in soup.find_all("a", href=True):
            full_url = urljoin(url, link["href"])

            if full_url not in visited and is_valid_url(full_url):
                to_visit.append(full_url)

        # Respekt politeness delay
        time.sleep(0.2)

    return results
