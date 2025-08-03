# chatbot/query_chat.py
import os
import re
from typing import Annotated, TypedDict, List, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# === API Key ===
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyBiVUeOqe12cgPIaphtiudVMEWyS9_mieo")

# === Load FAISS ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
base_dir = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(base_dir, "..", "data", "hou", "hou_index")

vectordb = FAISS.load_local(
    index_path,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True,
)

# === Gemini model ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.5,
    max_output_tokens=2048,
)

# === Prompt trả lời chính (RAG) ===
chat_main = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là một trợ lý AI thông minh, chỉ trả lời dựa trên dữ liệu đã được tinh chỉnh. "
        "Ưu tiên chính xác, logic, dễ hiểu, và chỉ dùng thông tin trong phần 'Tài liệu'."
        "với ngữ cảnh hội thoại (ví dụ: tên người dùng, tham chiếu như 'ngành này'), bạn được phép dùng lịch sử hội thoại"
        "Hãy TRÌNH BÀY KẾT QUẢ THEO MARKDOWN, ngắn gọn và rõ ràng, ưu tiên:\n"
        "1) Tiêu đề cấp 3 trở xuống (###),\n"
        "2) Danh sách gạch đầu dòng cho các ý chính,\n"
        "3) Bảng khi so sánh/ liệt kê theo cột,\n"
        "4) Chia đoạn ngắn; không viết thành một khối dài.\n"
        "Ưu tiên chính xác, logic, dễ hiểu luôn hoàn thành câu trọn vẹn.",
    ),
    MessagesPlaceholder(variable_name="messages"),
    ("user",
     "Dựa trên TÀI LIỆU dưới đây, hãy trả lời câu hỏi.\n\n"
     "TÀI LIỆU:\n{context}\n\n"
     "CÂU HỎI: {question}\n\nTrả lời:")
])
answer_chain = chat_main | llm

# === Prompt phân loại section ===
chat_section = ChatPromptTemplate.from_messages([
    (
        "system",
        "Bạn là bộ phân loại ý định. Hãy trả về đúng 1 từ trong: "
        "'giới thiệu', 'tuyển sinh', 'ngành học', 'sự kiện', 'thông báo', 'hợp tác', 'học phí', 'khác'. "
        "Không thêm ký tự thừa.",
    ),
    ("user", "Câu hỏi: {query}\nPhân loại:"),
])
section_chain = chat_section | llm

# === Heuristics nhận diện fact + năm ===
FACT_KEYWORDS = [
    "điểm sàn",
    "ngưỡng",
    "học phí",
    "ngày",
    "năm",
    "chỉ tiêu",
    "mã",
    "điểm",
    "thời hạn",
    "deadline",
    "tỷ lệ",
    "tỉ lệ",
    "bao nhiêu",
]
YEAR_RE = re.compile(r"(20\d{2})")

# === Khai báo STATE có bộ nhớ tin nhắn ===
class State(TypedDict):
    query: str
    section: str
    documents: List[Document]
    answer: str
    messages: Annotated[Sequence[BaseMessage], add_messages]  # [SỬA] cho phép auto-append



def last_user_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


def is_fact_query(query: str) -> bool:
    ql = query.lower()
    return any(k in ql for k in FACT_KEYWORDS) or bool(YEAR_RE.search(ql))


def year_priority_filter(query: str, docs: List[Document]) -> List[Document]:
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
    return with_year + others


def retrieve_with_threshold(query: str, min_score: float = 0.35, k_cap: int = 100) -> List[Document]:
    docs_scores = vectordb.similarity_search_with_relevance_scores(query, k=k_cap)
    filtered = [d for d, s in docs_scores if (s is None) or (s >= min_score)]
    if len(filtered) < 4:
        return [d for d, _ in docs_scores][:10]
    return filtered


def retrieve_with_mmr(query: str, section_value: str) -> List[Document]:
    try:
        return vectordb.max_marginal_relevance_search(
            query=query,
            k=100,
            fetch_k=100,
            lambda_mult=0.5,
            filter=lambda meta: section_value in meta.get("section", ""),
        )
    except Exception:
        return []

# === Node 1: phân loại + rút ra query từ messages ===

def classify_section(state: State) -> State:
    query = (state.get("query") or last_user_text(state.get("messages", [])) or "").strip()
    section = section_chain.invoke({"query": query}).content.strip().lower()
    if section not in [
        "giới thiệu",
        "tuyển sinh",
        "ngành học",
        "sự kiện",
        "thông báo",
        "hợp tác",
        "học phí",
        "khác",
    ]:
        section = "khác"
    print(f"📎 Gemini xác định section: {section}")
    return {**state, "section": section}

# === Node 2: retrieve ===

def retrieve_docs(state: State) -> State:
    query = state["query"]
    section_value = state["section"]

    if is_fact_query(query):
        docs = retrieve_with_threshold(query, min_score=0.35, k_cap=100)
        if len(docs) < 4:
            try:
                docs = vectordb.max_marginal_relevance_search(
                    query=query,
                    k=100,
                    fetch_k=100,
                    lambda_mult=0.5,
                    filter=lambda meta: section_value in meta.get("section", ""),
                )
            except Exception:
                docs = vectordb.max_marginal_relevance_search(
                    query=query, k=100, fetch_k=100, lambda_mult=0.5
                )
                print("⚠️ Threshold gắt hoặc không filter section, fallback MMR toàn bộ.")
    else:
        docs = retrieve_with_mmr(query, section_value)
        if not docs:
            docs = vectordb.max_marginal_relevance_search(
                query=query, k=100, fetch_k=100, lambda_mult=0.5
            )
            print("⚠️ Không filter theo section được, dùng MMR trên toàn bộ FAISS.")

    docs = year_priority_filter(query, docs)
    return {**state, "documents": docs}

# === Node 3: trả lời + đẩy AIMessage vào bộ nhớ ===

def truncate_docs(documents: List[Document], max_chars=4000) -> str:
    combined = ""
    for doc in documents:
        if len(combined) + len(doc.page_content) < max_chars:
            combined += doc.page_content + "\n\n"
        else:
            break
    return combined


def generate_answer(state: State) -> State:
    docs_text = truncate_docs(state["documents"], max_chars=4000)
    if not docs_text.strip():
        content = "Không tìm thấy thông tin liên quan."
        return {**state, "messages": [AIMessage(content=content)], "answer": content}

    try:
        # [SỬA] Truyền cả history
        resp = answer_chain.invoke({"context": docs_text, "messages": state["messages"],"question": state["query"],})
        content = (resp.content or "").strip() or "Không có thông tin phù hợp hoặc lỗi từ Gemini."
        return {**state, "messages": [AIMessage(content=content)], "answer": content}
    except Exception:
        content = "Không thể tạo câu trả lời do lỗi API."
        return {**state, "messages": [AIMessage(content=content)], "answer": content}



def add_user_message(state: State) -> State:
    return {**state, "messages": [HumanMessage(content=state["query"])]}

# === Xây graph có checkpointer (MemorySaver) ===
graph = StateGraph(State)

graph.add_node("add_user_message", add_user_message)

graph.add_node("classify", classify_section)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("answer", generate_answer)

graph.set_entry_point("add_user_message")
graph.add_edge("add_user_message", "classify")
graph.add_edge("classify", "retrieve")
graph.add_edge("retrieve", "answer")
graph.set_finish_point("answer")

# BẬT BỘ NHỚ: mỗi thread_id sẽ có lịch sử messages riêng
memory = MemorySaver()
chatbot = graph.compile(checkpointer=memory)

# === CLI test khi chạy trực tiếp ===
if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "demo"}}
    while True:
        q = input("❓ Hỏi: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        # [SỬA] gửi query để node add_user_message thêm HumanMessage
        result = chatbot.invoke({"query": q}, cfg)
        print("\n📌 Trả lời:", result.get("answer"))
