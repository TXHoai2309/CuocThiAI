import os
from typing import TypedDict, List
from click import prompt
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

# === Prompt trả lời chính ===
main_prompt = PromptTemplate.from_template("""Bạn là một trợ lý AI thông minh, trả lời câu hỏi dựa trên dữ liệu đã được tinh chỉnh. Hãy tổng hợp và cung cấp câu trả lời chính xác, logic, dễ hiểu và sáng tạo nếu cần dựa trên thông tin từ tài liệu sau:

**Tài liệu**: {context}

**Câu hỏi**: {question}

**Trả lời**:""")

# === Prompt phân loại section ===
section_prompt = PromptTemplate.from_template(
    """Phân tích câu hỏi sau và trả lời bằng đúng 1 từ trong: 
'giới thiệu', 'tuyển sinh', 'ngành học', 'sự kiện', 'thông báo', 'hợp tác', 'học phí', 'khác'.\n\nCâu hỏi: {query}\nPhân loại:"""
)
section_chain = section_prompt | llm

# === Khai báo trạng thái LangGraph ===
class QAState(TypedDict):
    query: str
    section: str
    documents: List[Document]
    answer: str

# === Node 1: Phân loại section ===
def classify_section(state: QAState) -> QAState:
    section = section_chain.invoke({"query": state["query"]}).content.strip().lower()
    if section not in ["giới thiệu", "tuyển sinh", "ngành học", "sự kiện", "thông báo", "hợp tác", "học phí", "khác"]:
        section = "khác"
    print(f"📎 Gemini xác định section: {section}")
    return {**state, "section": section}


# === Node 2: Truy xuất tài liệu từ FAISS ===
def retrieve_docs(state: QAState) -> QAState:
    # Thử tìm theo section nếu có metadata
    try:
        docs = vectordb.max_marginal_relevance_search(
            query=state["query"],
            k=100,
            fetch_k=100,
            lambda_mult=0.5,
            filter=lambda meta: state["section"] in meta.get("section", "")
        )
    except:
        docs = []  # nếu FAISS không hỗ trợ filter, fallback

    # Nếu không có kết quả, fallback: không dùng filter
    if not docs:
        docs = vectordb.max_marginal_relevance_search(
            query=state["query"],
            k=100,
            fetch_k=100,
            lambda_mult=0.5
        )
        print("⚠️ Không tìm được tài liệu theo section, dùng toàn bộ FAISS.")

    return {**state, "documents": docs}



# === Node 3: Tạo câu trả lời từ context ===
def truncate_docs(documents, max_chars=6000):
    combined = ""
    for doc in documents:
        if len(combined) + len(doc.page_content) < max_chars:
            combined += doc.page_content + "\n\n"
        else:
            break
    return combined

def generate_answer(state: QAState) -> QAState:
    docs_text = truncate_docs(state["documents"], max_chars=4000)  # Giảm max_chars
    print("📑 Context length:", len(docs_text))
    print("📑 Context sample:", docs_text[:500])
    if not docs_text.strip():
        print("⚠️ Empty context detected!")
        return {**state, "answer": "Không tìm thấy thông tin liên quan."}
    
    prompt = main_prompt.format(context=docs_text, question=state["query"])
    print("📝 Prompt sent to Gemini:", prompt[:500])
    
    try:
        response = llm.invoke(prompt)
        print("📤 Gemini raw response:", response)
        print("📤 Gemini response content:", response.content)
        if not response.content.strip():
            print("⚠️ Gemini returned empty response!")
            return {**state, "answer": "Không có thông tin phù hợp hoặc lỗi từ Gemini."}
    except Exception as e:
        print("❌ Gemini error:", str(e))
        return {**state, "answer": "Không thể tạo câu trả lời do lỗi API."}
    
    return {**state, "answer": response.content}



# === Xây LangGraph ===
graph = StateGraph(QAState)
graph.add_node("classify", classify_section)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("answer", generate_answer)

graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "answer")
graph.set_finish_point("answer")

chatbot = graph.compile()

# === Vòng lặp CLI ===
while True:
    query = input("❓ Hỏi: ").strip()
    if query.lower() in ["exit", "quit", "q"]:
        print("👋 Tạm biệt!")
        break

    result = chatbot.invoke({"query": query})
    print("\n📌 Trả lời:", result["answer"])
    print("\n📚 Từ các đoạn:")
    for doc in result["documents"]:
        print(f"- {doc.metadata.get('title', 'Không tiêu đề')} | {doc.page_content[:100]}...")
