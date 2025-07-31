from bs4 import BeautifulSoup

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def is_valid_url(url: str) -> bool:
    return url.startswith("http") and "hou.edu.vn" in url
