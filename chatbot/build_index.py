import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 📥 Bước 1: Load dữ liệu JSON (lùi 1 cấp từ chatbot ra CuocThiAI)
file_path = "menu_contents_refined.json"


print(f"📂 Đang mở file: {os.path.abspath(file_path)}")  # In đường dẫn tuyệt đối

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 📄 Bước 2: Chuyển đổi dữ liệu thành danh sách Document
documents = []
for item in data:
    if "content" in item and item["content"].strip():
        metadata = {
            "title": item.get("title"),
            "url": item.get("url"),
            "category": item.get("category"),
            "date": item.get("date"),
        }

        doc = Document(
            page_content=item["content"],
            metadata=metadata
        )
        documents.append(doc)

# 🤖 Bước 3: Tạo Embedding + FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents, embedding_model)

# 💾 Bước 4: Lưu FAISS index
output_path = os.path.join(os.path.dirname(__file__), "..", "data", "hou_index")
db.save_local(output_path)

print("✅ Đã lưu FAISS index vào:", os.path.abspath(output_path))
