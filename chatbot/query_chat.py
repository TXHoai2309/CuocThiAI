import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Cài đặt API key của Google
os.environ["GOOGLE_API_KEY"] = "AIzaSyC-AUp7NplKO6Y1RRtjwdSu6tRe2aqsknU"  # Thay bằng API key của bạn

# Đường dẫn đến file JSON chứa dữ liệu đã tinh chỉnh
JSON_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hou_crawler', 'crawler', 'data', 'menu_contents_refined.json'))

# Hàm tải dữ liệu từ file JSON
def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Lỗi khi tải file JSON: {e}")
        return None

# Tải FAISS vector store
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Đường dẫn tuyệt đối đến thư mục chứa index.faiss và index.pkl
    # Đường dẫn tuyệt đối tới THƯ MỤC chứa index.faiss và index.pkl
    INDEX_DIR = r"D:\airdrop\CuocThiAI\hou_crawler\crawler\data\hou_index"  # ✅ Đây là thư mục

    print(f"👉 Đang tải FAISS từ: {INDEX_DIR}")

    vectordb = FAISS.load_local(
        INDEX_DIR,  # ✅ Phải là thư mục, không phải file
        embeddings=embedding_model,
        index_name="index",  # Tương ứng với: index.faiss + index.pkl
        allow_dangerous_deserialization=True
    )


except Exception as e:
    print(f"Lỗi khi tải FAISS vector store: {e}") 
    exit(1)

# Khởi tạo mô hình Gemini
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Hoặc "gemini-1.5-pro" nếu bạn có quyền truy cập
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.5,
        max_output_tokens=512
    )
except Exception as e:
    print(f"Lỗi khi khởi tạo mô hình Gemini: {e}")
    exit(1)

# Tạo prompt template để cải thiện ngữ cảnh trả lời
prompt_template = """Bạn là một trợ lý AI thông minh, trả lời câu hỏi dựa trên dữ liệu đã được tinh chỉnh. Hãy cung cấp câu trả lời chính xác, ngắn gọn và dễ hiểu dựa trên thông tin từ tài liệu sau:

**Tài liệu**: {context}

**Câu hỏi**: {question}

**Trả lời**: """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Tạo QA chain với prompt tùy chỉnh
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"}
)

# Tải dữ liệu JSON để cung cấp ngữ cảnh bổ sung (nếu cần)
json_data = load_json_data(JSON_DATA_PATH)
if json_data is None:
    print("Không thể tiếp tục vì không tải được dữ liệu JSON.")
    exit(1)

# Vòng lặp hỏi đáp
while True:
    try:
        query = input("❓ Hỏi: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Thoát chương trình.")
            break
        response = qa_chain.invoke({"query": query})
        print("\n📌 Trả lời:", response["result"])
        # Hiển thị nguồn tài liệu (tùy chọn)
        print("\n📚 Nguồn tài liệu:")
        for doc in response["source_documents"]:
            print(f"- {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Lỗi khi xử lý câu hỏi: {e}")