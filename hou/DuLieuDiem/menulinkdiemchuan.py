import requests
from bs4 import BeautifulSoup

BASE_URL = "https://diemthi.tuyensinh247.com"

url = "https://diemthi.tuyensinh247.com/diem-chuan.html"
headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

schools = []
for li in soup.select("div.list-schol-box li"):
    a = li.find("a")
    if not a:
        continue
    href = a["href"]
    code_tag = a.find("strong")
    code = code_tag.text.strip() if code_tag else ""
    # Lấy tên trường (loại mã trường và dấu '-')
    name = a.get_text().replace(code, "").replace("-", "").strip()
    schools.append({
        "code": code,
        "name": name,
        "url": BASE_URL + href
    })

# In ra kết quả
for school in schools:
    print(f"{school['code']}\t{school['name']}\t{school['url']}")

# Nếu muốn lưu ra JSON:
import json
with open("school_links.json", "w", encoding="utf-8") as f:
    json.dump(schools, f, ensure_ascii=False, indent=2)
