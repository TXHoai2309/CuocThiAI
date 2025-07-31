import json
import requests
from bs4 import BeautifulSoup
import os

INPUT_PATH = "../data/menu_links.json"
OUTPUT_PATH = "../data/menu_contents.json"

def clean_html_text(html: str) -> str:
    """Làm sạch HTML, loại bỏ script/style và trả về text thuần"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def is_valid_url(url: str) -> bool:
    """Chỉ crawl các link thuộc domain hou.edu.vn"""
    return url.startswith("http") and "hou.edu.vn" in url

def extract_article_links_from_category(soup):
    """Lấy danh sách URL bài viết con trong trang category"""
    links = []
    for a in soup.select("article a"):
        href = a.get("href")
        if href and is_valid_url(href):
            links.append(href)
    return list(set(links))  # loại bỏ trùng

def crawl_single_article(url, category):
    """Crawl 1 bài viết chi tiết"""
    print(f"🌐 Đang crawl bài: {url}")
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        title_tag = soup.find('h1') or soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "Không rõ tiêu đề"

        date_tag = soup.find('time') or soup.find("div", class_="post-date")
        date = date_tag.get_text(strip=True) if date_tag else None

        article = (
            soup.find("div", class_="entry-content") or
            soup.find("div", class_="elementor-post__text") or
            soup.find("div", class_="elementor-widget-container") or
            soup.find("div", class_="post-content") or
            soup.find("article") or
            soup.find("body")
        )
        if not article:
            print(f"⚠️ Không tìm thấy nội dung phù hợp ở {url}")
            return None

        content_html = str(article)
        content_cleaned = clean_html_text(content_html)

        if not content_cleaned.strip():
            print(f"⚠️ Nội dung rỗng tại {url}, bỏ qua.")
            return None

        return {
            "title": title,
            "category": category,
            "url": url,
            "content_html": content_html,
            "content_cleaned": content_cleaned,
            "date": date
        }

    except Exception as e:
        print(f"❌ Lỗi khi crawl {url}: {e}")
        return None

def crawl_content_from_links(menu_links_path, output_path):
    with open(menu_links_path, 'r', encoding='utf-8') as f:
        links_data = json.load(f)

    results = []

    for item in links_data:
        url = item['url']
        category = item['category']

        if not is_valid_url(url):
            print(f"⏩ Bỏ qua URL không hợp lệ: {url}")
            continue

        try:
            print(f"🔗 Đang xử lý URL: {url}")
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')

            # Nếu là trang chứa nhiều bài viết (category)
            if soup.find_all("article"):
                article_links = extract_article_links_from_category(soup)
                print(f"📌 Phát hiện {len(article_links)} bài viết trong category {url}")
                for link in article_links:
                    article_data = crawl_single_article(link, category)
                    if article_data:
                        results.append(article_data)
            else:
                # Crawl như bài viết đơn
                article_data = crawl_single_article(url, category)
                if article_data:
                    results.append(article_data)

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {url}: {e}")

            # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 

    # Ghi file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Đã lưu {len(results)} bài viết vào: {output_path}")



if __name__ == "__main__":
    crawl_content_from_links(INPUT_PATH, OUTPUT_PATH)
