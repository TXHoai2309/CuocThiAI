# pdf_to_json.py
import os
import re
import json
import uuid
import argparse
from shutil import which
from typing import List, Dict, Optional

import pdfplumber
import pytesseract
from PIL import Image

# ====================== NHẬN DẠNG SECTION ======================
SECTION_PATTERNS = [
    r"^(A|B|C)\.\s+",
    r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)\.\s+",
    r"^I\.\d+\s", r"^II\.\d+\s", r"^III\.\d+\s",
    r"^MT\d+[:\s]?",
    r"^CDR\s*\d+[:\s]?",
    r"^(KHUNG CHƯƠNG TRÌNH|CHUẨN ĐẦU RA|MỤC TIÊU|THÔNG TIN TỔNG QUÁT|TỐT NGHIỆP)"
]
SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), flags=re.IGNORECASE)

# ====================== TIỆN ÍCH ======================
def ensure_tesseract(cmd_hint: Optional[str] = None) -> Optional[str]:
    """
    Tìm tesseract.exe theo thứ tự: đối số chỉ định, vị trí phổ biến, PATH hệ thống.
    Set sẵn pytesseract.tesseract_cmd + TESSDATA_PREFIX nếu tìm thấy.
    """
    candidates = [
        cmd_hint,
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        which("tesseract"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            pytesseract.pytesseract.tesseract_cmd = c
            os.environ.setdefault("TESSDATA_PREFIX", os.path.join(os.path.dirname(c), "tessdata"))
            return c
    return None

def is_heading(line: str) -> bool:
    line_norm = (line or "").strip()
    if not line_norm:
        return False
    if SECTION_REGEX.search(line_norm):
        return True
    letters = re.sub(r"[^A-ZÀ-Ỵ0-9 ]", "", line_norm.upper())
    return len(letters) >= 8 and letters == line_norm.upper()

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_tables(page) -> List[Dict]:
    tables = []
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
        "keep_blank_chars": True,
        "text_tolerance": 2,
        "join_tolerance": 2,
    }
    try:
        for t in page.extract_tables(settings):
            rows = []
            for r in t:
                row = [(c or "").strip() for c in r]
                if any(cell for cell in row):
                    rows.append(row)
            if rows:
                tables.append({"rows": rows})
    except Exception:
        pass
    return tables

def ocr_page(page, pageno: int, langs=("vie+eng", "vie", "eng"),
             resolution=350, dump_dir: Optional[str] = None) -> str:
    """OCR 1 trang với danh sách ngôn ngữ; lưu text ra dump_dir nếu có."""
    for lang in langs:
        try:
            im = page.to_image(resolution=resolution).original
            txt = pytesseract.image_to_string(im, lang=lang) or ""
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                with open(os.path.join(dump_dir, f"page_{pageno:02d}_{lang}.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(txt)
            if txt.strip():
                return txt
        except pytesseract.TesseractNotFoundError:
            raise
        except Exception:
            continue
    return ""

def extract_text_with_fallback(page, pageno: int, use_ocr: bool, lang: str,
                             verbose: bool = True, dump_dir: Optional[str] = None) -> str:
    # 1) Thử lấy layer text
    try:
        txt = page.extract_text() or ""
        if txt.strip():
            if verbose:
                print(f"[TXT] Trang {pageno}: {len(txt)} ký tự")
            return txt
    except Exception:
        pass

    # 2) OCR nếu bật
    if not use_ocr:
        if verbose:
            print(f"[SKIP OCR] Trang {pageno} không có layer text.")
        return ""

    if verbose:
        print(f"[OCR] Trang {pageno} → chạy OCR …")

    try:
        txt = ocr_page(page, pageno, langs=(lang, "vie", "eng"), dump_dir=dump_dir)
    except pytesseract.TesseractNotFoundError:
        raise RuntimeError(
            "Không tìm thấy Tesseract. Hãy cài đặt (https://github.com/UB-Mannheim/tesseract/wiki) "
            "và đảm bảo nó nằm trong PATH hệ thống. Hoặc sử dụng đối số --tesseract để "
            "chỉ định đường dẫn tesseract.exe."
        )
    if verbose:
        print(f"[OCR] Trang {pageno}: {len(txt)} ký tự sau OCR")
    return txt

# ====================== CORE ======================
def pdf_to_structured(pdf_path: str, ocr_lang: str = "vie+eng",
                      verbose: bool = True, use_ocr: bool = True,
                      tesseract_cmd: Optional[str] = None,
                      dump_dir: Optional[str] = None) -> Dict:
    tess_path = ensure_tesseract(tesseract_cmd)
    if use_ocr and not tess_path:
        raise RuntimeError(
            "Không tìm thấy Tesseract. Hãy cài đặt (https://github.com/UB-Mannheim/tesseract/wiki) "
            "và đảm bảo nó nằm trong PATH hệ thống. Hoặc sử dụng đối số --tesseract để "
            "chỉ định đường dẫn tesseract.exe."
        )

    out = {
        "metadata": {
            "source_file": os.path.basename(pdf_path),
            "source_path": os.path.abspath(pdf_path),
            "generator": "pdfplumber" + ("+Tesseract" if use_ocr and tess_path else ""),
            "schema_version": "v2.0",
            "pages": 0
        },
        "sections": []
    }

    with pdfplumber.open(pdf_path) as pdf:
        out["metadata"]["pages"] = len(pdf.pages)
        current = None
        temp_paragraphs = []

        for pageno, page in enumerate(pdf.pages, start=1):
            text = extract_text_with_fallback(
                page, pageno, use_ocr=use_ocr, lang=ocr_lang,
                verbose=verbose, dump_dir=dump_dir
            )

            lines = [l for l in (text.splitlines() if text else []) if l.strip()]
            tables = extract_tables(page)
            
            for line in lines:
                if is_heading(line):
                    if temp_paragraphs:
                        if current:
                            current["blocks"].append({
                                "type": "paragraph",
                                "page": pageno,
                                "text": clean_text("\n".join(temp_paragraphs))
                            })
                        temp_paragraphs = []
                    
                    if current:
                        current["page_end"] = pageno
                        out["sections"].append(current)
                    
                    current = {
                        "id": str(uuid.uuid4()),
                        "title": clean_text(line),
                        "page_start": pageno,
                        "page_end": pageno,
                        "blocks": []
                    }
                else:
                    if current is None:
                        current = {
                            "id": str(uuid.uuid4()),
                            "title": "PHẦN MỞ ĐẦU",
                            "page_start": pageno,
                            "page_end": pageno,
                            "blocks": []
                        }
                    temp_paragraphs.append(line)
            
            if temp_paragraphs:
                if current:
                    current["blocks"].append({
                        "type": "paragraph",
                        "page": pageno,
                        "text": clean_text("\n".join(temp_paragraphs))
                    })
                temp_paragraphs = []

            if tables:
                if current is None:
                    current = {
                        "id": str(uuid.uuid4()),
                        "title": "PHẦN MỞ ĐẦU",
                        "page_start": pageno,
                        "page_end": pageno,
                        "blocks": []
                    }
                for tb in tables:
                    current["blocks"].append({"type": "table", "page": pageno, "rows": tb["rows"]})

        if current:
            out["sections"].append(current)

    for sec in out["sections"]:
        sec["blocks"] = [b for b in sec["blocks"]
                         if not (b["type"] == "paragraph" and not b["text"])]
    return out

def save_json(data: Dict, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(out_path: str) -> Optional[Dict]:
    """Tải nội dung JSON từ file nếu tồn tại."""
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Cảnh báo: File JSON {out_path} bị lỗi, sẽ tạo file mới.")
            return None
    return None

def find_and_replace_file_data(data_list: list, new_data: Dict):
    """
    Tìm kiếm và thay thế dữ liệu của một file trong danh sách.
    Nếu không tìm thấy, thêm mới vào cuối.
    """
    file_name = new_data["metadata"]["source_file"]
    found = False
    for i, file_data in enumerate(data_list):
        if file_data.get("metadata", {}).get("source_file") == file_name:
            data_list[i] = new_data
            found = True
            break
    if not found:
        data_list.append(new_data)

# ====================== CLI ======================
def main():
    # === ĐỔI 2 ĐƯỜNG DẪN NÀY THEO MÁY BẠN NẾU CHẠY BẰNG TAY ===
    DEFAULT_PDF = r"D:\bai_tap_lon_cac_mon\CuocThiAI\hou_crawler\crawler\data\ChuongTrinhHoc\DTVT.pdf"
    DEFAULT_OUT = r"D:\bai_tap_lon_cac_mon\CuocThiAI\hou_crawler\crawler\data\ChuongTrinhHoc\Chuongtrinhhocc.json"

    ap = argparse.ArgumentParser(
        description="PDF → JSON có cấu trúc (sections/paragraph/table) với OCR dự phòng (Tesseract)."
    )
    ap.add_argument("--pdf", nargs='+', default=None,
                    help="Đường dẫn file PDF đầu vào (có thể là nhiều file, ví dụ: file1.pdf file2.pdf)")
    ap.add_argument("-o", "--out", default=None,
                    help="Đường dẫn file JSON đầu ra (mặc định dùng DEFAULT_OUT ở trên)")
    ap.add_argument("--lang", default="vie+eng",
                    help="Ngôn ngữ OCR cho Tesseract (vd: vie, eng, vie+eng). Mặc định: vie+eng")
    ap.add_argument("--no-ocr", action="store_true",
                    help="Tắt OCR fallback (nếu PDF là scan, sẽ không ra chữ).")
    ap.add_argument("--dump-ocr", default=None,
                    help="Thư mục xuất text OCR từng trang để debug (vd: ocr_dump)")
    ap.add_argument("--tesseract", default=None,
                    help="Đường dẫn đến file tesseract.exe (vd: C:\\Program Files\\Tesseract-OCR\\tesseract.exe)")
    ap.add_argument("--quiet", action="store_true", help="Ẩn log")
    args = ap.parse_args()

    # 1) Gán mặc định nếu thiếu
    pdf_paths = args.pdf or [DEFAULT_PDF]
    out_path = args.out or DEFAULT_OUT

    # 2) Chuẩn hoá & kiểm tra
    pdf_paths = [os.path.normpath(p) for p in pdf_paths]
    out_path = os.path.normpath(out_path)

    valid_pdf_paths = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"❌ Không tìm thấy file PDF: {pdf_path}. Bỏ qua.")
        else:
            valid_pdf_paths.append(pdf_path)

    if not valid_pdf_paths:
        print("❌ Không có file PDF hợp lệ nào để xử lý.")
        return

    # 3) Đọc file JSON hiện có nếu có, nếu không thì khởi tạo mới.
    combined_data = load_json(out_path)
    if combined_data is None:
        combined_data = {
            "metadata": {
                "generator": "pdfplumber+Tesseract",
                "schema_version": "v2.0"
            },
            "files": []
        }

    # 4) Chạy trích xuất và gom dữ liệu
    print(f"[INFO] OUT: {out_path}")
    print(f"[INFO] OCR lang: {args.lang} | use_ocr={not args.no_ocr}")

    try:
        for pdf_path in valid_pdf_paths:
            print(f"[INFO] Đang xử lý PDF: {pdf_path}")
            data = pdf_to_structured(
                pdf_path,
                ocr_lang=args.lang,
                verbose=not args.quiet,
                use_ocr=not args.no_ocr,
                tesseract_cmd=args.tesseract,
                dump_dir=args.dump_ocr,
            )
            # Tìm và thay thế hoặc thêm mới dữ liệu của file hiện tại
            find_and_replace_file_data(combined_data["files"], data)
            print(f"✅ Đã trích xuất và cập nhật dữ liệu từ: {os.path.basename(pdf_path)}")

        # 5) Ghi toàn bộ dữ liệu đã được cập nhật ra file JSON
        save_json(combined_data, out_path)
        print(f"✅ Đã lưu toàn bộ JSON có cấu trúc vào: {out_path}")
        
    except RuntimeError as e:
        print(f"❌ Lỗi: {e}")
        if "Tesseract" in str(e):
            print("\n💡 Hướng dẫn khắc phục:")
            print("1. **Cài đặt Tesseract:** Tải và cài đặt từ trang: https://github.com/UB-Mannheim/tesseract/wiki")
            print("2. **Thêm vào PATH:** Trong quá trình cài đặt, hãy đảm bảo bạn chọn tùy chọn 'Add to PATH'.")
            print("3. **Chỉ định đường dẫn:** Nếu đã cài nhưng không thêm vào PATH, hãy chạy lại lệnh với đối số --tesseract:")
            print(r"   Ví dụ: python pdf_to_json.py --tesseract 'C:\Program Files\Tesseract-OCR\tesseract.exe'")

if __name__ == "__main__":
    main()