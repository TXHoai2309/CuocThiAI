import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os

visited = set()
articles = []

def is_valid_url(url):
    """Chỉ lấy domain hou.edu.vn hoặc .hou.edu.vn"""
    try:
        domain = urlparse(url).netloc.lower()
        return domain == "hou.edu.vn" or domain.endswith(".hou.edu.vn")
    except:
        return False

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def crawl_article(url, category_path, depth=0):
    indent = "  " * depth
    if url in visited:
        return
    visited.add(url)
    try:
        print(f"{indent}📝 {url}")
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")

        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Không rõ tiêu đề"

        date_tag = soup.find("time") or soup.find("div", class_="post-date")
        date = date_tag.get_text(strip=True) if date_tag else None

        content_div = (
            soup.find("div", class_="entry-content") or
            soup.find("div", class_="post-content") or
            soup.find("article") or
            soup.find("body")
        )

        if not content_div:
            print(f"{indent}⚠️ Không tìm thấy nội dung.")
            return

        content_html = str(content_div)
        content_cleaned = clean_html_text(content_html)

        if not content_cleaned.strip():
            print(f"{indent}⚠️ Nội dung rỗng. Bỏ qua.")
            return

        article_data = {
            "title": title,
            "url": url,
            "date": date,
            "category_path": category_path,
            "content_html": content_html,
            "content_cleaned": content_cleaned
        }

        articles.append(article_data)

    except Exception as e:
        print(f"{indent}❌ Lỗi bài viết: {e}")

def crawl_tree(url, category_name, parent_path, depth=0):
    indent = "  " * depth
    if url in visited or not is_valid_url(url):
        return
    visited.add(url)
    print(f"{indent}📂 {url}")

    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)

        for a in links:
            href = a['href'].strip()
            title = a.get_text(strip=True)
            if not is_valid_url(href) or href in visited:
                continue

            if "category" not in href and "page" not in href and len(href) > 20:
                crawl_article(href, parent_path + [category_name], depth + 1)
            else:
                crawl_tree(href, title, parent_path + [category_name], depth + 1)

    except Exception as e:
        print(f"{indent}❌ Lỗi menu: {e}")

def main():
    INPUT = "../data/menu_links.json"
    OUTPUT_ARTICLES = "../data/menu_contents.json"

    with open(INPUT, 'r', encoding='utf-8') as f:
        menu_links = json.load(f)

    for item in menu_links:
        root_url = item["url"]
        root_name = item["category"]
        if is_valid_url(root_url) and root_url not in visited:
            print(f"\n🌐 Bắt đầu từ: {root_name}")
            crawl_tree(root_url, root_name, [], depth=0)

    os.makedirs(os.path.dirname(OUTPUT_ARTICLES), exist_ok=True)

    with open(OUTPUT_ARTICLES, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu {len(articles)} bài viết tại: {OUTPUT_ARTICLES}")

if __name__ == "__main__":
    main()
