# build_index_diem_chuan.py
import json
import os
from collections import Counter
from typing import List, Dict, Any, Iterable
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ====== CẤU HÌNH ĐƯỜNG DẪN (CHỈNH TÙY MÔI TRƯỜNG) ======
# Ví dụ: INPUT_FILE = "/mnt/data/all_schools_diem_chuan_2024.json"
INPUT_FILE = r"D:\airdrop\CuocThiAI\HOU\DuLieuDiem\all_schools_diem_chuan_2024.json"

# Thư mục lưu FAISS index (đặt cạnh file hiện tại -> ./data/diem_chuan_index)
OUTPUT_INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "data", "diem_chuan_index")
)


# ====== HÀM TIỆN ÍCH ======
def _safe_str(x):
    return "" if x is None else str(x)


def _normalize_method(m: str) -> str:
    """
    Chuẩn hoá tên phương thức để đồng nhất metadata/section.
    Gom các biến thể 'ĐGNL HN/HCM/SPHN', 'thi riêng', 'xét tuyển kết hợp',...
    """
    if not m:
        return "khác"
    s = m.lower()

    # Các nhóm chính
    if "thpt" in s:
        return "điểm thi thpt"
    if "học bạ" in s or "hoc ba" in s:
        return "điểm học bạ"
    if "đgnl" in s or "dg nl" in s or "dgnl" in s or "đánh giá năng lực" in s:
        return "điểm đgnl"  # gom HN/HCM/SPHN
    if "tư duy" in s or "tu duy" in s or "đánh giá tư duy" in s:
        return "điểm đánh giá tư duy"
    if "thi riêng" in s or "kỳ thi riêng" in s or "ky thi rieng" in s:
        return "điểm thi riêng"
    if "xét tuyển kết hợp" in s or "xet tuyen ket hop" in s:
        return "điểm xét tuyển kết hợp"

    # Mặc định giữ nguyên dạng thường
    return s


def _normalize_combos(raw_combos: Iterable[Any]) -> List[str]:
    """
    Chuẩn hoá danh sách tổ hợp môn.
    - Một số dữ liệu có phần tử dạng "A00, A01" -> tách theo dấu phẩy.
    - Loại bỏ khoảng trắng dư thừa, phần tử rỗng.
    """
    out: List[str] = []
    for c in (raw_combos or []):
        for t in _safe_str(c).split(","):
            t = t.strip()
            if t:
                out.append(t)
    # Khử trùng lặp, giữ thứ tự xuất hiện
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _iter_school_blocks(root_obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Chuẩn hoá thành iterable các khối “trường”.
    - Nếu gốc là list: duyệt trực tiếp.
    - Nếu gốc là object: bọc vào list để duyệt.
    """
    if isinstance(root_obj, list):
        return root_obj
    return [root_obj]


# ====== LOAD & CHUYỂN ĐỔI JSON -> Document ======
def load_documents(json_path: str) -> List[Document]:
    """
    Kỳ vọng cấu trúc hiện tại:
    [
      {
        "school_code": "...",
        "school_name": "...",
        "source_url": "...",
        "collected_at": "...",
        "records": [
           {
             "ten_nganh": "...",
             "to_hop_mon": ["A00","A01", ...] hoặc ["A00, A01", ...],
             "diem_chuan": 23.5,
             "ghi_chu": "...",
             "_method": "Điểm thi THPT",
             "_year": 2024
           },
           ...
        ]
      },
      ...
    ]

    (Cũng hỗ trợ định dạng cũ dạng object đơn.)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        root = json.load(f)

    docs: List[Document] = []
    school_counter = 0
    record_counter = 0

    for block in _iter_school_blocks(root):
        school_counter += 1
        source_url = _safe_str(block.get("source_url"))
        school_name = _safe_str(block.get("school_name") or block.get("school"))
        school_code = _safe_str(block.get("school_code"))
        collected_at = _safe_str(block.get("collected_at"))
        records = block.get("records", [])

        for rec in records:
            record_counter += 1
            major = _safe_str(rec.get("ten_nganh"))
            combos = _normalize_combos(rec.get("to_hop_mon"))
            combos_str = ", ".join(combos) if combos else ""
            score = rec.get("diem_chuan")
            note = _safe_str(rec.get("ghi_chu"))
            method_raw = _safe_str(rec.get("_method"))
            method = _normalize_method(method_raw)
            year = rec.get("_year")

            # Nội dung chính để truy vấn (page_content)
            content_lines = [
                f"Trường: {school_name or school_code}",
                f"Năm: {year}" if year else "",
                f"Ngành: {major}",
                f"Tổ hợp môn: {combos_str}" if combos_str else "",
                f"Điểm chuẩn: {score}",
                f"Phương thức: {method_raw}" if method_raw else "",
                f"Ghi chú: {note}" if note else "",
                f"Nguồn: {source_url}" if source_url else "",
            ]
            page_content = "\n".join([line for line in content_lines if line.strip()])

            doc_id_school = (school_code or school_name or "unknown").lower()
            metadata = {
                "school": school_name or school_code,
                "school_code": school_code,
                "year": year,
                "major": major,
                "combos": combos,          # list chuẩn hoá
                "score": score,
                "method": method,          # đã chuẩn hoá
                "method_raw": method_raw,  # giữ nguyên bản gốc
                "note": note,
                "source_url": source_url,
                "collected_at": collected_at,
                # Section giúp phân nhóm
                "section": f"điểm chuẩn · {method}" if method else "điểm chuẩn",
                # id gợi ý (duy nhất theo: school-year-major-method)
                "doc_id": f"{doc_id_school}|{year}|{major}|{method}".lower(),
            }

            docs.append(Document(page_content=page_content, metadata=metadata))

    if school_counter == 0:
        print("⚠️  Không tìm thấy khối 'trường' nào trong file JSON.")
    else:
        print(f"🔎 Thống kê: {school_counter} trường, {record_counter} bản ghi.")
    return docs


# ====== MAIN PIPELINE ======
def main():
    # 1) Kiểm tra file đầu vào
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Không tìm thấy INPUT_FILE: {INPUT_FILE}\n"
            f"Hãy chỉnh lại biến INPUT_FILE cho đúng đường dẫn."
        )

    print("📥 Đang tải và xử lý dữ liệu điểm chuẩn...")
    documents = load_documents(INPUT_FILE)
    print(f"✅ Tổng số Document tạo ra: {len(documents)}")

    if not documents:
        print("⚠️  Không có Document nào được tạo. Dừng.")
        return

    # 2) Thống kê nhanh (tuỳ chọn)
    methods = Counter([d.metadata.get("method") for d in documents])
    years = Counter([d.metadata.get("year") for d in documents])
    print("📊 Phân bố phương thức:", dict(methods))
    print("📆 Phân bố năm:", dict(years))

    # 3) Chia nhỏ tài liệu (mỗi record thường ngắn, chunk nhỏ & ít overlap)
    print("🔪 Chia nhỏ tài liệu (nếu cần)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(documents)
    print(f"📄 Tổng số đoạn sau chia: {len(split_docs)}")

    # 4) Nhúng & build FAISS
    print("🧠 Đang nhúng dữ liệu với HuggingFaceEmbeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(split_docs, embedding_model)

    # 5) Lưu index
    os.makedirs(OUTPUT_INDEX_DIR, exist_ok=True)
    print(f"💾 Lưu FAISS index vào: {OUTPUT_INDEX_DIR}")
    vectordb.save_local(OUTPUT_INDEX_DIR)

    print("🎉 Hoàn tất build index điểm chuẩn!")


if __name__ == "__main__":
    main()
