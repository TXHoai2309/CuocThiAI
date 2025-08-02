# chatbot/query_chat.py
import os
import re
from typing import TypedDict, List, Sequence
import secrets
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from langchain.chains import LLMChain


# === API Key ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyDR1eVkKtTN3RBeXNdW3bThRIwMMMfJND8"

# === Load FAISS ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

base_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(base_dir, "..", "hou_crawler", "crawler", "data", "hou_index")

vectordb = FAISS.load_local(
    index_path,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True
)

# === Gemini model ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.5,
    max_output_tokens=2048
)

# === Prompt trả lời chính ===
chat_main = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là một trợ lý AI thông minh, chỉ trả lời dựa trên dữ liệu đã được tinh chỉnh. "
     "Ưu tiên chính xác, logic, dễ hiểu, và chỉ dùng thông tin trong phần 'Tài liệu'."
     "Luôn hoàn thành câu trọn vẹn, không bỏ dở từ/câu."),
    # Đưa context vào dưới vai trò system để mô hình xem đây là nguồn sự thật
    ("system", "Tài liệu (context):\n{context}"),
    ("user", "{question}")
])
answer_chain = chat_main | llm  # pipe: prompt -> llm

# === Prompt phân loại section ===
chat_section = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là bộ phân loại ý định. Hãy trả về đúng 1 từ trong: "
     "'giới thiệu', 'tuyển sinh', 'ngành học', 'sự kiện', 'thông báo', 'hợp tác', 'học phí', 'khác'. "
     "Không thêm ký tự thừa."),
    ("user", "Câu hỏi: {query}\nPhân loại:")
])
section_chain = chat_section | llm  # pipe: prompt -> llm


# === Heuristics nhận diện câu hỏi fact + năm ===
FACT_KEYWORDS = [
    "điểm sàn", "ngưỡng", "học phí", "ngày", "năm",
    "chỉ tiêu", "mã", "điểm", "thời hạn", "deadline", "tỷ lệ", "tỉ lệ", "bao nhiêu"
]
YEAR_RE = re.compile(r"(20\d{2})")

# === Khai báo trạng thái LangGraph ===
class QAState(TypedDict):
    query: str
    section: str
    documents: List[Document]
    answer: str

def is_fact_query(query: str) -> bool:
    """
    Mục đích: Nhận diện câu hỏi dạng fact/số liệu (có năm, ngày, con số...)
    -> dùng retriever kiểu THRESHOLD để siết nhiễu cho chính xác hơn.
    """
    ql = query.lower()
    return any(keyword in ql for keyword in FACT_KEYWORDS) or bool(YEAR_RE.search(ql))

def year_priority_filter(query: str, docs: List[Document]) -> List[Document]:
    """
    Mục đích: Nếu query có năm (20xx), ưu tiên các doc có năm đó trong
    content hoặc metadata (title/url/date/section/category).
    """
    m = YEAR_RE.search(query)
    if not m:
        return docs
    year = m.group(1)

    def has_year(d: Document) -> bool:
        meta = d.metadata or {}
        hay = (d.page_content or "") + " " + " ".join(
            str(meta.get(k, "")) for k in ["date", "title", "url", "section", "category"]
        )
        return year in hay

    with_year = [d for d in docs if has_year(d)]
    others = [d for d in docs if not has_year(d)]
    return with_year + others  # ưu tiên khớp-năm nằm trước

def retrieve_with_threshold(query: str, min_score: float = 0.35, k_cap: int = 100) -> List[Document]:
    """
    Retriever kiểu 'lọc theo ngưỡng' để tăng precision cho câu hỏi fact.

    - Gọi similarity_search_with_relevance_scores -> trả (Document, score).
    - Lọc những doc có score >= min_score.
    - Fallback: nếu sau lọc còn quá ít (<4) -> trả top 10 ban đầu (giữ recall).
    """
    docs_scores = vectordb.similarity_search_with_relevance_scores(query, k=k_cap)
    filtered = [d for d, s in docs_scores if (s is None) or (s >= min_score)]
    if len(filtered) < 4:
        return [d for d, _ in docs_scores][:10]
    return filtered

# MỤC ĐÍCH: Truy xuất kiểu MMR = tăng đa dạng, giảm trùng lặp (hữu ích cho câu hỏi khái quát)
def retrieve_with_mmr(query: str, section_value: str) -> List[Document]:
    """
    Ưu tiên filter theo 'section' (nếu metadata có) để giảm nhiễu.
    Nếu backend không hỗ trợ filter lambda -> trả [] để node retrieve_docs fallback.
    """
    try:
        return vectordb.max_marginal_relevance_search(
            query=query,
            k=100,
            fetch_k=100,
            lambda_mult=0.5,
            filter=lambda meta: section_value in meta.get("section", "")
        )
    except Exception:
        return []  # để node retrieve_docs biết fallback

# === Node 1: Phân loại section ===
def classify_section(state: QAState) -> QAState:
    section = section_chain.invoke({"query": state["query"]}).content.strip().lower()
    if section not in ["giới thiệu", "tuyển sinh", "ngành học", "sự kiện", "thông báo", "hợp tác", "học phí", "khác"]:
        section = "khác"
    print(f"📎 Gemini xác định section: {section}")
    return {**state, "section": section}

# === Node 2: Truy xuất tài liệu từ FAISS ===
def retrieve_docs(state: QAState) -> QAState:
    """
    MỤC ĐÍCH NODE (đã chỉnh nhẹ):
    1) Câu hỏi FACT -> dùng THRESHOLD để SIẾT NHIỄU (tăng chính xác).
       - Nếu kết quả quá ít -> Fallback sang MMR để giữ độ bao phủ.
    2) Câu hỏi KHÁI QUÁT -> dùng MMR để ĐA DẠNG ngữ cảnh (giảm trùng lặp).
    3) Sau khi có danh sách docs -> ƯU TIÊN THEO NĂM (nếu query có năm).
       (ĐÃ BỎ bước boost theo metadata như bạn yêu cầu.)
    """
    query = state["query"]
    section_value = state["section"]

    # 1) Chọn chiến lược theo loại câu hỏi
    if is_fact_query(query):
        docs = retrieve_with_threshold(query, min_score=0.35, k_cap=100)
        if len(docs) < 4:  # fallback để không bị thiếu ngữ cảnh
            try:
                docs = vectordb.max_marginal_relevance_search(
                    query=query, k=100, fetch_k=100, lambda_mult=0.5,
                    filter=lambda meta: section_value in meta.get("section", "")
                )
            except Exception:
                docs = vectordb.max_marginal_relevance_search(
                    query=query, k=100, fetch_k=100, lambda_mult=0.5
                )
                print("⚠️ Threshold quá gắt hoặc không filter được theo section, fallback MMR toàn bộ FAISS.")
    else:
        # Câu hỏi khái quát -> MMR (giảm trùng lặp)
        docs = retrieve_with_mmr(query, section_value)
        if not docs:
            # Fallback nếu filter lambda không chạy được
            docs = vectordb.max_marginal_relevance_search(
                query=query, k=100, fetch_k=100, lambda_mult=0.5
            )
            print("⚠️ Không filter theo section được, dùng MMR trên toàn bộ FAISS.")

    # 2) Ưu tiên các đoạn cùng NĂM với query (nếu có)
    docs = year_priority_filter(query, docs)

    return {**state, "documents": docs}

# === Node 3: Tạo câu trả lời từ context ===
def truncate_docs(documents, max_chars=6000):
    """
    Mục đích: gom các đoạn context đủ ngắn (<= max_chars) để nhét vào prompt,
    tránh tràn token. Giữ nguyên thứ tự đã chọn.
    """
    combined = ""
    for doc in documents:
        if len(combined) + len(doc.page_content) < max_chars:
            combined += doc.page_content + "\n\n"
        else:
            break
    return combined

def generate_answer(state: QAState) -> QAState:
    docs_text = truncate_docs(state["documents"], max_chars=4000)  # ngưỡng 4k ký tự để giảm rủi ro tràn token
    if not docs_text.strip():
        print("⚠️ Empty context detected!")
        return {**state, "answer": "Không tìm thấy thông tin liên quan."}

    # Gọi chain chuẩn: truyền {context, question} vào prompt -> LLM
    try:
        resp = answer_chain.invoke({"context": docs_text, "question": state["query"]})
        content = (resp.content or "").strip()
        if not content:
            return {**state, "answer": "Không có thông tin phù hợp hoặc lỗi từ Gemini."}
        return {**state, "answer": content}
    except Exception as e:
        print("❌ Gemini error:", str(e))
        return {**state, "answer": "Không thể tạo câu trả lời do lỗi API."}

# === Xây LangGraph ===
graph = StateGraph(QAState)
graph.add_node("classify", classify_section)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("answer", generate_answer)

graph.set_entry_point("classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "answer")
graph.set_finish_point("answer")

# ... các phần trên giữ nguyên ...

chatbot = graph.compile()

# === Vòng lặp CLI chỉ chạy khi chạy trực tiếp file này ===
if __name__ == "__main__":
    while True:
        query = input("❓ Hỏi: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Tạm biệt!")
            break
        result = chatbot.invoke({"query": query})
        print("\n📌 Trả lời:", result["answer"])
