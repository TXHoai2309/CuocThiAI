import json
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ====== CẤU HÌNH ĐƯỜNG DẪN ======
# Đường dẫn file JSON điểm chuẩn (đã upload)
INPUT_FILE = r"D:\airdrop\CuocThiAI\hou_crawler\crawler\data\DuLieuDiem\diem_chuan_MHN_2024_tuyensinh247.json"

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
    Ví dụ: 'Điểm thi THPT', 'Điểm học bạ', 'Điểm ĐGNL HN', 'Điểm Đánh giá Tư duy'
    """
    if not m:
        return "khác"
    m_lower = m.lower()
    if "thpt" in m_lower:
        return "điểm thi thpt"
    if "học bạ" in m_lower:
        return "điểm học bạ"
    if "đgnl" in m_lower:
        return "điểm đgnl hn"
    if "tư duy" in m_lower:
        return "điểm đánh giá tư duy"
    return m_lower

# ====== LOAD & CHUYỂN ĐỔI JSON -> Document ======
def load_documents(json_path: str) -> List[Document]:
    """
    Kỳ vọng cấu trúc:
    {
      "source_url": "...",
      "school": "...",
      "collected_at": "...",
      "records": [
         {
           "ten_nganh": "...",
           "to_hop_mon": ["A00","D01",...],
           "diem_chuan": 23.5,
           "ghi_chu": "...",
           "_method": "Điểm thi THPT",
           "_year": 2024
         },
         ...
      ]
    }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    source_url = _safe_str(payload.get("source_url"))
    school = _safe_str(payload.get("school"))
    collected_at = _safe_str(payload.get("collected_at"))
    records = payload.get("records", [])

    docs: List[Document] = []
    for i, rec in enumerate(records):
        major = _safe_str(rec.get("ten_nganh"))
        combos = rec.get("to_hop_mon") or []
        combos_str = ", ".join([_safe_str(x) for x in combos]) if combos else ""
        score = rec.get("diem_chuan")
        note = _safe_str(rec.get("ghi_chu"))
        method_raw = _safe_str(rec.get("_method"))
        method = _normalize_method(method_raw)
        year = rec.get("_year")

        # Nội dung chính để truy vấn (page_content)
        content_lines = [
            f"Trường: {school}" if school else "",
            f"Năm: {year}" if year else "",
            f"Ngành: {major}",
            f"Tổ hợp môn: {combos_str}" if combos_str else "",
            f"Điểm chuẩn: {score}",
            f"Phương thức: {method_raw}" if method_raw else "",
            f"Ghi chú: {note}" if note else "",
            f"Nguồn: {source_url}" if source_url else "",
        ]
        page_content = "\n".join([line for line in content_lines if line.strip()])

        # Metadata phục vụ lọc nhanh
        metadata = {
            "school": school,
            "year": year,
            "major": major,
            "combos": combos,          # list
            "score": score,
            "method": method,          # đã chuẩn hoá
            "method_raw": method_raw,  # giữ nguyên bản gốc
            "note": note,
            "source_url": source_url,
            "collected_at": collected_at,
            # Section giúp phân nhóm
            "section": f"điểm chuẩn · {method}" if method else "điểm chuẩn",
            # id gợi ý (duy nhất theo: school-year-major-method)
            "doc_id": f"{school}|{year}|{major}|{method}".lower(),
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    return docs

# ====== MAIN PIPELINE ======
def main():
    print("📥 Đang tải và xử lý dữ liệu điểm chuẩn...")
    documents = load_documents(INPUT_FILE)
    print(f"✅ Tổng số bản ghi: {len(documents)}")

    # Với dữ liệu điểm chuẩn, mỗi record khá ngắn -> chunk nhỏ, ít/không chồng lấn
    print("🔪 Chia nhỏ tài liệu (nếu cần)...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    split_docs = splitter.split_documents(documents)
    print(f"📄 Tổng số đoạn sau chia: {len(split_docs)}")

    print("🧠 Đang nhúng dữ liệu với HuggingFaceEmbeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = FAISS.from_documents(split_docs, embedding_model)

    # Tạo thư mục nếu chưa có
    os.makedirs(OUTPUT_INDEX_DIR, exist_ok=True)

    print(f"💾 Lưu FAISS index vào: {OUTPUT_INDEX_DIR}")
    vectordb.save_local(OUTPUT_INDEX_DIR)

    print("🎉 Hoàn tất build index điểm chuẩn!")

if __name__ == "__main__":
    main()
