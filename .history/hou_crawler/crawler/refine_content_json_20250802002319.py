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
visited_urls = set()

# ==== Selenium Chrome cấu hình headless ====
options = Options()
options.add_argument('--headless')  # Ẩn trình duyệt
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
    domain = parsed.netloc
    return "hou.edu.vn" in domain and all(excl not in domain for excl in EXCLUDE_DOMAINS)

def get_full_content(url: str) -> str:
    """Truy cập trang và lấy nội dung đầy đủ (nếu có nút Xem thêm)"""
    try:
        driver.get(url)
        time.sleep(2)  # chờ trang tải

        try:
            # Click vào nút "Xem thêm" nếu có
            btn = driver.find_element(By.XPATH, "//a[contains(text(), 'Xem thêm') or contains(text(), 'Read more')]")
            btn.click()
            time.sleep(1)
        except NoSuchElementException:
            pass  # Không có nút, bỏ qua

        # Tìm các vùng nội dung phổ biến
        try:
            content_div = driver.find_element(By.CLASS_NAME, "single-content-full")
        except NoSuchElementException:
            try:
                content_div = driver.find_element(By.TAG_NAME, "article")
            except NoSuchElementException:
                try:
                    content_div = driver.find_element(By.CLASS_NAME, "entry-content")
                except NoSuchElementException:
                    return ""

        return content_div.text.strip()

    except Exception as e:
        print(f"❌ Lỗi khi lấy nội dung từ {url}: {e}")
        return ""

# ==== Xử lý refine ====

def refine_json(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    refined_data = []

    for i, item in enumerate(raw_data):
        url = item.get("url", "").strip()
        if not url or url in visited_urls or not is_allowed_domain(url):
            continue

        visited_urls.add(url)
        print(f"🔍 ({i + 1}/{len(raw_data)}) Đang xử lý: {url}")

        content = get_full_content(url)
        if not content.strip():
            print(f"⚠️ Nội dung rỗng tại {url}, bỏ qua.")
            continue

        date = item.get("date") or extract_date(content)

        refined_data.append({
            "title": item.get("title", ""),
            "url": url,
            "category": item.get("category", ""),
            "date": date,
            "content": content
        })

    # Lưu ra file JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu file refined: {output_path} (Tổng {len(refined_data)} bài)")

# ==== Chạy chính ====
if __name__ == "__main__":
    refine_json(INPUT_FILE, OUTPUT_FILE)
    driver.quit()
