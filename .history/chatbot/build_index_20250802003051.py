import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Đường dẫn file JSON đầu vào
INPUT_FILE =  r"D:\bai_tap_lon_cac_mon\CuocThiAI\hou_crawler\crawler\data\hou_index"
OUTPUT_INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hou_crawler', 'crawler', 'data', 'hou_index'))

# Tải và chuyển đổi dữ liệu thành list Document
def load_documents(json_path):
    documents = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        content = item.get("content", "").strip()
        if not content:
            continue

        title = (item.get("title") or "").lower()
        url = (item.get("url") or "").lower()
        category = (item.get("category") or "").lower()

        # Gán section theo logic đơn giản
        if any(x in title for x in ["giới thiệu", "lịch sử", "sứ mệnh", "tầm nhìn", "triết lý"]):
            section = "giới thiệu"
        elif "tuyển sinh" in title or "tuyển sinh" in url:
            section = "tuyển sinh"
        elif any(x in title or x in category for x in ["ngành", "chuyên ngành", "chương trình"]):
            section = "ngành học"
        elif "sự kiện" in category or "event" in url:
            section = "sự kiện"
        elif "học phí" in category or "học phí" in title:
            section = "học phí"
        elif "hợp tác" in category or "hợp tác" in title:
            section = "hợp tác"
        elif "thông báo" in category or "thông báo" in title:
            section = "thông báo"
        else:
            section = "khác"

        doc = Document(
            page_content=content,
            metadata={
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "category": item.get("category", ""),
                "section": section
            }
        )
        documents.append(doc)

    return documents

def main():
    print("📥 Đang tải và xử lý dữ liệu JSON...")
    documents = load_documents(INPUT_FILE)

    print("🔪 Chia nhỏ tài liệu...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_docs = splitter.split_documents(documents)

    print(f"📄 Tổng số đoạn sau chia: {len(split_docs)}")
    
    print("🧠 Đang nhúng dữ liệu với HuggingFaceEmbeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embedding_model)

    print(f"💾 Lưu FAISS index vào thư mục: {OUTPUT_INDEX_DIR}")
    vectordb.save_local(OUTPUT_INDEX_DIR)
    print("✅ Hoàn tất!")

if __name__ == "__main__":
    main()
