import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import os

visited = set()
articles_new = []      # bài mới trong lần chạy
existing_urls = set()  # URL đã có trong file kết quả

# ===== Logger có đánh số thứ tự dòng =====
LOG_IDX = 1
def log(msg: str):
    """In ra console với số thứ tự 1., 2., ... trước mỗi dòng."""
    global LOG_IDX
    print(f"{LOG_IDX}. {msg}")
    LOG_IDX += 1

def contains_hou(url: str) -> bool:
    """Chấp nhận mọi URL miễn là có chứa 'hou' (không phân biệt hoa/thường)."""
    try:
        return "hou" in url.lower()
    except:
        return False

def is_crawlable(url: str) -> bool:
    """Loại bỏ scheme không hợp lệ."""
    if not url:
        return False
    low = url.lower()
    if low.startswith(("mailto:", "javascript:", "#")):
        return False
    return True

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)

def number_lines(text: str) -> str:
    """Đánh số 1., 2., ... trước mỗi dòng không rỗng; giữ dòng trống nếu có."""
    lines = text.splitlines()
    out, idx = [], 1
    for line in lines:
        if line.strip():
            out.append(f"{idx}. {line}")
            idx += 1
        else:
            out.append("")
    return "\n".join(out)

def crawl_article(url, category_path, depth=0):
    indent = "  " * depth

    if url in visited:
        log(f"{indent}🔁 Bỏ qua (đã thăm): {url}")
        return

    # Bỏ qua nếu URL đã có trong dữ liệu trước đó
    if url in existing_urls:
        visited.add(url)
        log(f"{indent}⏭️ Bỏ qua (đã có trong dữ liệu): {url}")
        return

    visited.add(url)
    try:
        log(f"{indent}📝 Bắt đầu crawl bài: {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
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
            log(f"{indent}⚠️ Không tìm thấy nội dung: {url}")
            return

        content_html = str(content_div)
        content_cleaned = clean_html_text(content_html)
        if not content_cleaned.strip():
            log(f"{indent}⚠️ Nội dung rỗng, bỏ qua: {url}")
            return

        # Đánh số thứ tự trước mỗi dòng nội dung
        content_numbered = number_lines(content_cleaned)

        article_data = {
            "title": title,
            "url": url,
            "date": date,
            "category_path": category_path,
            "content_html": content_html,
            "content_cleaned": content_numbered
        }

        articles_new.append(article_data)
        existing_urls.add(url)  # tránh trùng trong cùng phiên
        log(f"{indent}✅ Đã crawl xong: {title} | {url}")

    except Exception as e:
        log(f"{indent}❌ Lỗi khi crawl bài: {url} | {e}")

def crawl_tree(url, category_name, parent_path, depth=0):
    indent = "  " * depth

    if url in visited:
        log(f"{indent}🔁 Bỏ qua node (đã thăm): {url}")
        return
    if not is_crawlable(url):
        log(f"{indent}⛔ Bỏ qua (không crawlable): {url}")
        return
    if not contains_hou(url):
        log(f"{indent}⛔ Bỏ qua (không chứa 'hou'): {url}")
        return

    visited.add(url)
    log(f"{indent}📂 Duyệt menu: {url}")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)

        for a in links:
            href_raw = a['href'].strip()
            if not is_crawlable(href_raw):
                log(f"{indent}  ⛔ Bỏ qua link không hợp lệ: {href_raw}")
                continue

            href = urljoin(resp.url, href_raw)  # chuẩn hóa absolute URL

            if not contains_hou(href):
                log(f"{indent}  ⛔ Bỏ qua (không chứa 'hou'): {href}")
                continue
            if href in visited:
                log(f"{indent}  🔁 Bỏ qua (đã thăm): {href}")
                continue

            title = a.get_text(strip=True)

            # Heuristic phân biệt bài viết / menu
            if "category" not in href and "page" not in href and len(href) > 20:
                log(f"{indent}  ➜ Phát hiện bài viết: {href}")
                crawl_article(href, parent_path + [category_name], depth + 1)
            else:
                log(f"{indent}  ➜ Đi sâu menu: {href}")
                crawl_tree(href, title or category_name, parent_path + [category_name], depth + 1)

    except Exception as e:
        log(f"{indent}❌ Lỗi khi duyệt menu: {url} | {e}")

def main():
    INPUT = "../data/menu_links.json"
    OUTPUT_ARTICLES = "../data/menu_content.json"  # theo yêu cầu

    # Tải dữ liệu đã có (nếu tồn tại) để không crawl lại
    existing_articles = []
    if os.path.exists(OUTPUT_ARTICLES):
        try:
            with open(OUTPUT_ARTICLES, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
            for it in existing_articles:
                if isinstance(it, dict) and "url" in it:
                    existing_urls.add(it["url"])
            log(f"📦 Nạp dữ liệu cũ: {len(existing_articles)} bài.")
        except Exception as e:
            log(f"⚠️ Không đọc được dữ liệu cũ, sẽ tạo lại: {e}")
            existing_articles = []
            existing_urls.clear()

    # Nạp menu link gốc
    with open(INPUT, 'r', encoding='utf-8') as f:
        menu_links = json.load(f)

    for item in menu_links:
        root_url = item.get("url", "").strip()
        root_name = item.get("category", "").strip() or "ROOT"
        if not is_crawlable(root_url):
            log(f"⛔ Bỏ qua root không crawlable: {root_url}")
            continue
        if not contains_hou(root_url):
            log(f"⛔ Bỏ qua root (không chứa 'hou'): {root_url}")
            continue
        log(f"\n🌐 Bắt đầu từ: {root_name} | {root_url}")
        crawl_tree(root_url, root_name, [], depth=0)

    # Hợp nhất dữ liệu cũ + mới (ghi đè URL trùng bằng bản mới)
    merged_by_url = {}
    for it in existing_articles:
        if isinstance(it, dict) and "url" in it:
            merged_by_url[it["url"]] = it
    for it in articles_new:
        merged_by_url[it["url"]] = it  # ghi đè để cập nhật nội dung mới

    merged_list = list(merged_by_url.values())

    os.makedirs(os.path.dirname(OUTPUT_ARTICLES), exist_ok=True)
    with open(OUTPUT_ARTICLES, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    log(f"\n✅ Đã lưu tổng cộng {len(merged_list)} bài viết tại: {OUTPUT_ARTICLES}")
    log(f"➕ Mới thêm trong lần chạy này: {len(articles_new)}")

if __name__ == "__main__":
    main()
