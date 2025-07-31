import json
import re
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "data", "menu_contents.json")
OUTPUT_REFINED_FILE = os.path.join(BASE_DIR, "data", "menu_contents_refined.json")

# 👉 Cấu hình Selenium Chrome headless
options = Options()
options.add_argument('--headless')  # Ẩn trình duyệt (bỏ dòng này nếu muốn hiện)
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def extract_date(content):
    """Trích xuất ngày từ nội dung nếu có dạng dd/mm/yyyy"""
    match = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", content)
    return match.group(1) if match else None

def get_full_content(url):
    """Dùng Selenium để lấy toàn bộ nội dung, click nút Read more nếu có"""
    try:
        driver.get(url)
        time.sleep(2)  # chờ tải trang

        # Click nút Read more hoặc Xem thêm nếu có
        try:
            btn = driver.find_element(By.XPATH, "//a[contains(text(), 'Read more') or contains(text(), 'Xem thêm')]")
            btn.click()
            time.sleep(1)
        except NoSuchElementException:
            pass

        # Tìm div chứa nội dung chính
        try:
            content_div = driver.find_element(By.CLASS_NAME, "single-content-full")
            return content_div.text.strip()
        except NoSuchElementException:
            return ""
    except Exception as e:
        print(f"❌ Lỗi lấy nội dung {url}: {e}")
        return ""

def refine_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    refined = []
    for i, item in enumerate(data):
        url = item.get("url", "")
        print(f"🔍 ({i+1}/{len(data)}) Đang xử lý: {url}")

        content = get_full_content(url)
        if not content:
            continue

        # Nếu không có date, lấy từ content
        date = item.get("date") or extract_date(content)

        refined.append({
            "title": item.get("title", ""),
            "url": url,
            "category": item.get("category", ""),
            "date": date,
            "content": content
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu file đã lọc: {output_path} (Tổng {len(refined)} bài)")

if __name__ == "__main__":
    refine_json(INPUT_FILE, OUTPUT_REFINED_FILE)
    driver.quit()
