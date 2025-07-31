import requests
from bs4 import BeautifulSoup
import json
import os


def crawl_menu(url="https://hou.edu.vn", output_path="../data/menu_links.json"):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    menu_ul = soup.find("ul", class_="menu")
    if not menu_ul:
        print("❌ Không tìm thấy thẻ <ul class='menu'>")
        return

    menu_items = menu_ul.find_all("li", recursive=True)

    raw_links = []
    for item in menu_items:
        a_tag = item.find("a")
        if not a_tag:
            continue
        href = a_tag.get("href", "").strip()
        title = a_tag.text.strip()
        if href and href.startswith("http") and not href.startswith("#"):
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

    print(f"✅ Đã lưu {len(unique_links)} menu link vào {output_path}")


if __name__ == "__main__":
    crawl_menu()
