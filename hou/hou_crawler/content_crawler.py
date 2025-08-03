import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag, urlunparse
import os

# ================== Trạng thái toàn cục ==================
visited = set()         # chứa URL đã chuẩn hoá (canonical)
articles_new = []       # bài mới trong lần chạy
existing_urls = set()   # URL đã có trong file kết quả (canonical)

# ===== Logger có đánh số thứ tự dòng =====
LOG_IDX = 1

def log(msg: str):
    """In ra console với số thứ tự 1., 2., ... trước mỗi dòng."""
    global LOG_IDX
    print(f"{LOG_IDX}. {msg}")
    LOG_IDX += 1

# ================== Tiện ích URL & bộ lọc ==================

def canon(url: str, keep_query: bool = True) -> str:
    """
    Chuẩn hoá URL để so trùng.
    - Bỏ #fragment
    - GIỮ query có ý nghĩa (page, id, ...) nhưng loại tracking: utm_*, fbclid, gclid, msclkid, spm, ref, referrer, source
    - Hạ chữ thường scheme/host
    - Bỏ 'www.' nếu có
    - Bỏ '/' cuối nếu không phải root
    """
    if not url:
        return ""
    try:
        u, _ = urldefrag(url.strip())
        p = urlparse(u)
        scheme = (p.scheme or "http").lower()
        netloc = (p.netloc or "").lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path or "/"
        if path.endswith("/") and path != "/":
            path = path[:-1]

        query = ""
        if keep_query and p.query:
            from urllib.parse import parse_qsl, urlencode
            tracking_prefixes = ("utm_",)
            tracking_names = {"fbclid", "gclid", "msclkid", "spm", "ref", "referrer", "source"}
            pairs = []
            for k, v in parse_qsl(p.query, keep_blank_values=True):
                lk = (k or "").lower()
                if lk in tracking_names:
                    continue
                if any(lk.startswith(pref) for pref in tracking_prefixes):
                    continue
                pairs.append((k, v))
            if pairs:
                query = urlencode(pairs, doseq=True)

        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return url.strip()


def normalize_url(href: str, base_url: str) -> str:
    """Tuyệt đối hoá & trả về bản canonical của URL (giữ query hữu ích)."""
    abs_url = urljoin(base_url, (href or "").strip())
    return canon(abs_url, keep_query=True)


def is_http_https(url: str) -> bool:
    try:
        scheme = urlparse(url).scheme.lower()
        return scheme in ("http", "https")
    except Exception:
        return False


def skip_youtube(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return ("youtube.com" in host) or ("youtu.be" in host)
    except Exception:
        return False


def contains_hou(url: str) -> bool:
    """Chỉ chấp nhận URL có chữ 'hou' (không phân biệt hoa/thường)."""
    try:
        return "hou" in (url or "").lower()
    except Exception:
        return False


def is_crawlable(url: str) -> bool:
    """Loại bỏ scheme không hợp lệ/JS/mail/fragment."""
    if not url:
        return False
    low = url.lower()
    if low.startswith(("mailto:", "javascript:", "#")):
        return False
    return True

# ================== Xử lý nội dung ==================

def clean_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "iframe"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def number_lines(text: str) -> str:
    """Đánh số 1., 2., ... trước mỗi dòng không rỗng; giữ dòng trống nếu có."""
    lines = (text or "").splitlines()
    out, idx = [], 1
    for line in lines:
        if line.strip():
            out.append(f"{idx}. {line}")
            idx += 1
        else:
            out.append("")
    return "\n".join(out)

# ================== Crawl bài viết ==================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 HouCrawler/1.0"
}


def crawl_article(url: str, category_path, depth=0):
    indent = "  " * depth
    cu = canon(url, keep_query=True)

    if cu in visited:
        log(f"{indent}🔁 Bỏ qua (đã thăm): {cu}")
        return

    # Bỏ qua nếu URL đã có trong dữ liệu trước đó
    if cu in existing_urls:
        visited.add(cu)
        log(f"{indent}⏭️ Bỏ qua (đã có trong dữ liệu): {cu}")
        return

    visited.add(cu)
    try:
        log(f"{indent}📝 Bắt đầu crawl bài: {cu}")
        resp = requests.get(cu, headers=HEADERS, timeout=20)
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
            log(f"{indent}⚠️ Không tìm thấy nội dung: {cu}")
            return

        content_html = str(content_div)
        content_cleaned = clean_html_text(content_html)
        if not content_cleaned.strip():
            log(f"{indent}⚠️ Nội dung rỗng, bỏ qua: {cu}")
            return

        content_numbered = number_lines(content_cleaned)

        article_data = {
            "title": title,
            "url": cu,                    # canonical (giữ query)
            "final_url": canon(resp.url, keep_query=True),
            "date": date,
            "category_path": category_path,
            "content_html": content_html,
            "content_cleaned": content_numbered,
        }

        articles_new.append(article_data)
        existing_urls.add(cu)  # tránh trùng trong cùng phiên
        log(f"{indent}✅ Đã crawl xong: {title} | {cu}")

    except Exception as e:
        log(f"{indent}❌ Lỗi khi crawl bài: {cu} | {e}")

# ================== Crawl theo cây ==================

def crawl_tree(url: str, category_name: str, parent_path, depth=0):
    indent = "  " * depth
    cu = canon(url, keep_query=True)

    if cu in visited:
        log(f"{indent}🔁 Bỏ qua node (đã thăm): {cu}")
        return
    if not is_crawlable(cu):
        log(f"{indent}⛔ Bỏ qua (không crawlable): {cu}")
        return
    if not is_http_https(cu):
        log(f"{indent}⛔ Bỏ qua (không phải http/https): {cu}")
        return
    if skip_youtube(cu):
        log(f"{indent}⛔ Bỏ qua (YouTube): {cu}")
        return
    if not contains_hou(cu):
        log(f"{indent}⛔ Bỏ qua (không chứa 'hou'): {cu}")
        return

    visited.add(cu)
    log(f"{indent}📂 Duyệt menu: {cu}")

    try:
        resp = requests.get(cu, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            raw = a['href'].strip()
            if not is_crawlable(raw):
                continue

            href = normalize_url(raw, resp.url)  # absolute + canonical

            if not is_http_https(href) or skip_youtube(href) or not contains_hou(href):
                continue
            if href in visited:
                continue

            title = a.get_text(strip=True) or category_name

            # Heuristic phân biệt bài viết / menu
            if "category" not in href and "page" not in href and len(href) > 20:
                log(f"{indent}  ➜ Phát hiện bài viết: {href}")
                crawl_article(href, parent_path + [category_name], depth + 1)
            else:
                log(f"{indent}  ➜ Đi sâu menu: {href}")
                crawl_tree(href, title, parent_path + [category_name], depth + 1)

    except Exception as e:
        log(f"{indent}❌ Lỗi khi duyệt menu: {cu} | {e}")

# ================== main ==================

def main():
    INPUT = "../data/menu_links.json"
    OUTPUT_ARTICLES = "../data/menu_content.json"  # theo yêu cầu

    # Tải dữ liệu đã có (nếu tồn tại) để không crawl lại & KHÔNG ghi đè
    existing_articles = []
    if os.path.exists(OUTPUT_ARTICLES):
        try:
            with open(OUTPUT_ARTICLES, 'r', encoding='utf-8') as f:
                existing_articles = json.load(f)
            for it in existing_articles:
                if isinstance(it, dict):
                    u = it.get("url") or it.get("final_url")
                    if u:
                        existing_urls.add(canon(u, keep_query=True))
            log(f"📦 Nạp dữ liệu cũ: {len(existing_articles)} bài.")
        except Exception as e:
            log(f"⚠️ Không đọc được dữ liệu cũ, sẽ tạo lại: {e}")
            existing_articles = []
            existing_urls.clear()

    # Nạp menu link gốc
    with open(INPUT, 'r', encoding='utf-8') as f:
        menu_links = json.load(f)

    for item in menu_links:
        root_url = canon((item.get("url") or "").strip(), keep_query=True)
        root_name = (item.get("category") or "").strip() or "ROOT"
        if not is_crawlable(root_url):
            log(f"⛔ Bỏ qua root không crawlable: {root_url}")
            continue
        if not is_http_https(root_url):
            log(f"⛔ Bỏ qua root (không phải http/https): {root_url}")
            continue
        if skip_youtube(root_url):
            log(f"⛔ Bỏ qua root (YouTube): {root_url}")
            continue
        if not contains_hou(root_url):
            log(f"⛔ Bỏ qua root (không chứa 'hou'): {root_url}")
            continue

        log(f"\n🌐 Bắt đầu từ: {root_name} | {root_url}")
        crawl_tree(root_url, root_name, [], depth=0)

    # HỢP NHẤT: KHÔNG ghi đè dữ liệu đã có — chỉ thêm bài mới
    existing_set = {canon((it.get("url") or it.get("final_url") or ""), keep_query=True)
                    for it in existing_articles if isinstance(it, dict)}

    unique_new = []
    for it in articles_new:
        cu = canon(it.get("url") or it.get("final_url") or "", keep_query=True)
        if cu and cu not in existing_set:
            unique_new.append(it)
            existing_set.add(cu)

    merged_list = existing_articles + unique_new

    os.makedirs(os.path.dirname(OUTPUT_ARTICLES), exist_ok=True)
    with open(OUTPUT_ARTICLES, 'w', encoding='utf-8') as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    log(f"\n✅ Đã lưu tổng cộng {len(merged_list)} bài viết tại: {OUTPUT_ARTICLES}")
    log(f"➕ Mới thêm trong lần chạy này: {len(unique_new)} (đã bỏ qua bài trùng)")


if __name__ == "__main__":
    main()
