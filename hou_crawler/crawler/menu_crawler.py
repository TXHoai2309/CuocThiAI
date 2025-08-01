import requests
from bs4 import BeautifulSoup
import json
import os

def crawl_menu(url="https://hou.edu.vn", output_path="../data/menu_links.json"):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Lỗi khi kết nối: {e}")
        return

    soup = BeautifulSoup(res.text, "html.parser")

    # Tìm tất cả các thẻ <a> nằm trong <ul class="menu"> kể cả lồng nhau
    menu_ul = soup.find("ul", class_="menu")
    if not menu_ul:
        print("❌ Không tìm thấy thẻ <ul class='menu'>")
        return

    a_tags = menu_ul.find_all("a", href=True)

    raw_links = []
    for a in a_tags:
        href = a["href"].strip()
        title = a.get_text(strip=True)
        if not href or href.startswith("#") or "javascript:void" in href:
            continue
        if not href.startswith("http"):
            href = url.rstrip("/") + "/" + href.lstrip("/")
        raw_links.append({
            "category": title,
            "url": href
        })

    # Lọc trùng URL
    seen = set()
    unique_links = []
    for link in raw_links:
        if link["url"] not in seen:
            unique_links.append(link)
            seen.add(link["url"])

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_links, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã lưu {len(unique_links)} đường link menu vào {output_path}")

if __name__ == "__main__":
    crawl_menu()
