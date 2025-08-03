import json
import os
import re
import time
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

# === Load danh sách trường ===
with open(r"D:\bai_tap_lon_cac_mon\CuocThiAI\hou\DuLieuDiem\school_links.json", encoding="utf-8") as f:
    schools = json.load(f)

OUTDIR = "diem_chuan_all"  # Thư mục lưu các file json kết quả
os.makedirs(OUTDIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}

def norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_table(table, method_label: str, year: int) -> List[Dict]:
    headers = []
    thead = table.find("thead")
    if thead:
        for th in thead.find_all(["th", "td"]):
            headers.append(norm_text(th.get_text()))
    else:
        first_tr = table.find("tr")
        if first_tr:
            for th in first_tr.find_all(["th", "td"]):
                headers.append(norm_text(th.get_text()))
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
            header_map[h] = re.sub(r"\W+", "_", h_low).strip("_")
    rows_parent = table.find("tbody") or table
    records = []
    for tr in rows_parent.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = {}
        for i, cell in enumerate(cells):
            val = norm_text(cell.get_text())
            if i < len(headers):
                raw_h = headers[i]
                key = header_map.get(raw_h, f"col_{i}")
            else:
                key = f"col_{i}"
            row[key] = val
        # Chuẩn hóa các trường quan trọng
        if "to_hop_mon" in row:
            row["to_hop_mon"] = [x.strip() for x in row["to_hop_mon"].split(";") if x.strip()]
        if "diem_chuan" in row:
            dc = row["diem_chuan"].replace(",", ".")
            try:
                row["diem_chuan"] = float(dc)
            except:
                row["diem_chuan"] = row["diem_chuan"]
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
    title = norm_text(h3_tag.get_text())
    m = re.search(r"phương thức\s+(.*?)\s+năm", title, flags=re.I | re.U)
    if m:
        return norm_text(m.group(1))
    return title

def crawl_school(school):
    url = school['url']
    code = school['code']
    name = school['name']
    print(f"Cào: {code} - {name} - {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as ex:
        print(f"  Lỗi khi tải: {url}\n  {ex}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    year_guess = 2024  # Có thể cập nhật logic lấy năm tự động nếu muốn
    results = []
    for h3 in soup.find_all(["h2", "h3"]):
        heading = norm_text(h3.get_text())
        if "Điểm chuẩn theo phương thức" in heading:
            method_label = extract_method_label(h3)
            node = h3.find_next(lambda tag: tag.name in ["table", "div"] and (
                tag.name == "table" or
                ("ant-table" in " ".join(tag.get("class", [])) or tag.get("role") == "table")
            ))
            table = None
            if node:
                if node.name == "table":
                    table = node
                else:
                    inner = node.find("table")
                    if inner:
                        table = inner
            if not table:
                table = h3.find_next("table")
            if table:
                records = parse_table(table, method_label=method_label, year=year_guess)
                results.extend(records)

    # --- KHÔNG phân chia theo tổ hợp môn/khối nữa, chỉ 1 array duy nhất ---

    data_out = {
        "source_url": url,
        "school_code": code,
        "school_name": name,
        "section": code,    # section là mã trường, hoặc có thể để name nếu muốn
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": results  # Toàn bộ list điểm chuẩn
    }
    out_file = os.path.join(OUTDIR, f"diem_chuan_{code}_2024_tuyensinh247.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_out, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu {out_file} ({len(results)} records)")
    return {
        "source_url": url,
        "school_code": code,
        "school_name": name,
        "section": code,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": results
    }


if __name__ == "__main__":
    all_schools = []
    for i, school in enumerate(schools, 1):
        print(f"[{i}/{len(schools)}]", end=" ")
        data = crawl_school(school)
        if data:
            all_schools.append(data)
        time.sleep(1)  # tránh bị block

    # Sau khi xong, ghi ra 1 file JSON chung:
    out_file = "all_schools_diem_chuan_2024.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_schools, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu {len(all_schools)} trường vào {out_file}")