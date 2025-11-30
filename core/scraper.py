import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.comune.arezzo.it"

def crawl_comune_arezzo():
    r = requests.get(BASE_URL)
    soup = BeautifulSoup(r.text, "html.parser")
    return [{
        "url": BASE_URL,
        "title": soup.title.string if soup.title else "Homepage",
        "text": soup.get_text(" ", strip=True)
    }]
