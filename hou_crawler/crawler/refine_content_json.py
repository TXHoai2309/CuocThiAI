import json
import re
import os
import time
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# ==== Cấu hình ====
BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "data", "menu_contents.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "menu_contents_refined.json")

EXCLUDE_DOMAINS = ["jshou.edu.vn", "thuvien.hou.edu.vn", "sinhvien.hou.edu.vn"]

# ==== Selenium Chrome cấu hình headless ====
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# ==== Tiện ích ====

def extract_date(content: str) -> str:
    """Trích xuất ngày theo định dạng dd/mm/yyyy"""
    match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", content)
    return match.group(1) if match else None

def is_allowed_domain(url: str) -> bool:
    """Chỉ cho phép domain thuộc hou.edu.vn và không nằm trong danh sách loại trừ"""
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower()
    return "hou.edu.vn" in domain and all(excl not in domain for excl in EXCLUDE_DOMAINS)

def get_full_content(url: str) -> str:
    """Truy cập trang và lấy nội dung đầy đủ (nếu có nút Xem thêm/Read more)"""
    try:
        driver.get(url)
        time.sleep(2)
        try:
            btn = driver.find_element(By.XPATH, "//a[contains(text(), 'Xem thêm') or contains(text(), 'Read more')]")
            btn.click()
            time.sleep(1)
        except NoSuchElementException:
            pass

        # Tìm các vùng nội dung phổ biến
        for by, value in [
            (By.CLASS_NAME, "single-content-full"),
            (By.TAG_NAME, "article"),
            (By.CLASS_NAME, "entry-content"),
        ]:
            try:
                content_div = driver.find_element(by, value)
                return content_div.text.strip()
            except NoSuchElementException:
                continue

        return ""
    except Exception as e:
        print(f"❌ Lỗi khi lấy nội dung từ {url}: {e}")
        return ""

# ==== Xử lý refine ====

def refine_json(input_path: str, output_path: str):
    # 1) Nạp danh sách nguồn (thô)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2) Nạp dữ liệu refined cũ nếu có, để chỉ crawl URL mới
    existing_by_url = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                old_refined = json.load(f)
            for doc in old_refined:
                url = (doc.get("url") or "").strip()
                if url:
                    existing_by_url[url] = doc
            print(f"📦 Đã nạp {len(existing_by_url)} bài từ dữ liệu refined cũ.")
        except Exception as e:
            print(f"⚠️ Không đọc được dữ liệu refined cũ, sẽ ghi mới toàn bộ: {e}")
            existing_by_url = {}

    new_refined = []
    total = len(raw_data)
    processed = 0
    added = 0

    # 3) Duyệt các URL từ file thô, chỉ làm với URL chưa có trong refined
    for i, item in enumerate(raw_data, start=1):
        url = (item.get("url") or "").strip()
        if not url or not is_allowed_domain(url):
            continue

        processed += 1
        if url in existing_by_url:
            print(f"⏭️ ({i}/{total}) Bỏ qua (đã có trong refined): {url}")
            continue

        print(f"🔍 ({i}/{total}) Xử lý mới: {url}")
        content = get_full_content(url)
        if not content.strip():
            print(f"⚠️ Nội dung rỗng tại {url}, bỏ qua.")
            continue

        # Lấy ngày: ưu tiên 'date' có sẵn, sau đó trích từ content
        date_val = item.get("date") or extract_date(content)

        # Lấy category: ưu tiên 'category', fallback từ 'category_path'
        category_val = item.get("category")
        if not category_val:
            cp = item.get("category_path")
            if isinstance(cp, list):
                category_val = " > ".join(cp)
            else:
                category_val = ""

        new_refined.append({
            "title": item.get("title", ""),
            "url": url,
            "category": category_val,
            "date": date_val,
            "content": content
        })
        added += 1

    # 4) Hợp nhất: dữ liệu cũ + các bản mới (vì chỉ crawl URL mới nên không đè bản cũ)
    merged_by_url = dict(existing_by_url)
    for doc in new_refined:
        merged_by_url[doc["url"]] = doc  # thêm bản mới

    merged_list = list(merged_by_url.values())

    # 5) Ghi đè file refined
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu refined: {output_path}")
    print(f"• Nguồn tổng: {total}")
    print(f"• Bỏ qua (đã có): {len(existing_by_url)}")
    print(f"• Mới thêm: {added}")
    print(f"• Tổng sau hợp nhất: {len(merged_list)}")

# ==== Chạy chính ====
if __name__ == "__main__":
    try:
        refine_json(INPUT_FILE, OUTPUT_FILE)
    finally:
        driver.quit()
