import json
import re
import time
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

URL = "https://diemthi.tuyensinh247.com/diem-chuan/dai-hoc-mo-ha-noi-MHN.html"
OUTFILE = "diem_chuan_MHN_2024_tuyensinh247.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}

# Chuẩn hóa text: strip + thay nhiều khoảng trắng -> 1 space
def norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_table(table, method_label: str, year: int) -> List[Dict]:
    # Lấy header
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all(["th", "td"]):
            headers.append(norm_text(th.get_text()))
    else:
        # Một số trang render bảng không có <thead> chuẩn
        first_tr = table.find("tr")
        if first_tr:
            for th in first_tr.find_all(["th", "td"]):
                headers.append(norm_text(th.get_text()))

    # Map header -> key chuẩn
    header_map = {}
    for h in headers:
        h_low = h.lower()
        if "stt" in h_low:
            header_map[h] = "stt"
        elif "mã ngành" in h_low or "ma nganh" in h_low:
            header_map[h] = "ma_nganh"
        elif "tên ngành" in h_low or "ten nganh" in h_low:
            header_map[h] = "ten_nganh"
        elif "tổ hợp" in h_low or "to hop" in h_low:
            header_map[h] = "to_hop_mon"
        elif "điểm chuẩn" in h_low or "diem chuan" in h_low:
            header_map[h] = "diem_chuan"
        elif "ghi chú" in h_low or "ghi chu" in h_low:
            header_map[h] = "ghi_chu"
        else:
            # cột lạ -> giữ nguyên
            header_map[h] = re.sub(r"\W+", "_", h_low).strip("_")

    # Lấy body rows
    rows_parent = table.find("tbody") or table
    records = []
    for tr in rows_parent.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = {}
        for i, cell in enumerate(cells):
            val = norm_text(cell.get_text())
            # Xác định header cho ô này
            if i < len(headers):
                raw_h = headers[i]
                key = header_map.get(raw_h, f"col_{i}")
            else:
                key = f"col_{i}"
            row[key] = val

        # Chuẩn hóa các trường quan trọng
        # to_hop: list
        if "to_hop_mon" in row:
            row["to_hop_mon"] = [x.strip() for x in row["to_hop_mon"].split(";") if x.strip()]

        # điểm: float nếu có
        if "diem_chuan" in row:
            dc = row["diem_chuan"].replace(",", ".")
            try:
                row["diem_chuan"] = float(dc)
            except:
                row["diem_chuan"] = row["diem_chuan"]  # giữ nguyên chuỗi

        # stt: int nếu có
        if "stt" in row:
            try:
                row["stt"] = int(re.sub(r"\D", "", row["stt"]))
            except:
                pass

        row["_method"] = method_label
        row["_year"] = year
        records.append(row)

    return records

def extract_method_label(h3_tag) -> str:
    """
    Ví dụ tiêu đề: 'Điểm chuẩn theo phương thức Điểm thi THPT năm 2024'
    -> method_label = 'Điểm thi THPT'
    """
    title = norm_text(h3_tag.get_text())
    # cố gắng rút gọn phần giữa 'phương thức ' và ' năm'
    m = re.search(r"phương thức\s+(.*?)\s+năm", title, flags=re.I | re.U)
    if m:
        return norm_text(m.group(1))
    # fallback: trả toàn bộ title
    return title

def scrape():
    resp = requests.get(URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Tìm tất cả các h3 tiêu đề phần + bảng ngay sau đó
    results = []
    year_guess = 2024  # Theo trang hiện tại (đến thời điểm viết)
    for h3 in soup.find_all(["h2", "h3"]):
        heading = norm_text(h3.get_text())
        if "Điểm chuẩn theo phương thức" in heading:
            method_label = extract_method_label(h3)

            # Lấy bảng gần nhất sau tiêu đề
            node = h3.find_next(lambda tag: tag.name in ["table", "div"] and (
                tag.name == "table" or
                ("ant-table" in " ".join(tag.get("class", [])) or tag.get("role") == "table")
            ))

            # Nếu là div ant-table, tìm <table> bên trong
            table = None
            if node:
                if node.name == "table":
                    table = node
                else:
                    inner = node.find("table")
                    if inner:
                        table = inner

            if not table:
                # fallback: thử tìm table tiếp theo trong DOM
                table = h3.find_next("table")

            if table:
                records = parse_table(table, method_label=method_label, year=year_guess)
                results.extend(records)

    data_out = {
        "source_url": URL,
        "school": "Trường Đại học Mở Hà Nội (HOU)",
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": results
    }

    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu {len(results)} dòng vào {OUTFILE}")

if __name__ == "__main__":
    scrape()
