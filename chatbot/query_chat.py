# chatbot/query_chat.py
import os
import re
from typing import Annotated, TypedDict, List, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# === API Key ===
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyDR1eVkKtTN3RBeXNdW3bThRIwMMMfJND8")

# === Load FAISS ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
base_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến index FAISS
hou_index_path = os.path.join(base_dir, "..", "data", "hou", "hou_index")
diem_index_path = os.path.join(base_dir, "..", "data", "hou", "diem_chuan_index")

# Tải index FAISS với embeddings
vectordb_hou = FAISS.load_local(
    hou_index_path,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True,
)

# [GIỮ NGUYÊN THEO CODE CỦA BẠN]
# Lưu ý: index_name ("diem_index") phải TRÙNG với lúc build FAISS cho kho điểm.
vectordb_diem = FAISS.load_local(
    diem_index_path,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True,
)

# [SỬA] Danh sách các kho được dùng cho lần truy vấn hiện tại (router sẽ cập nhật)
# Lý do: Cho phép chọn HOU/Điểm/cả hai mà KHÔNG đổi tên hàm retrieve gốc.
selected_vector_stores: List[FAISS] = [vectordb_hou]

# === Gemini model ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
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

# === Heuristics: regex năm để ưu tiên tài liệu chứa năm ===
YEAR_RE = re.compile(r"(20\d{2})")

# --- Router keywords ---
DIEM_KEYWORDS = ["điểm", "điểm chuẩn", "điểm sàn", "ngưỡng", "tổ hợp", "mã ngành", "chỉ tiêu"]
HOU_KEYWORDS = [
    "ngành", "chương trình", "học phí", "thông báo", "sự kiện", "tuyển sinh",
    "giới thiệu", "hợp tác", "khoa", "viện", "môn", "học phần", "hồ sơ"
]

# === Khai báo STATE có bộ nhớ tin nhắn ===
class State(TypedDict):
    query: str
    section: str
    documents: List[Document]
    answer: str
    messages: Annotated[Sequence[BaseMessage], add_messages]  # cho phép auto-append



def last_user_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return (m.content or "").strip()
    return ""


# [SỬA] BỎ `FACT_KEYWORDS` và `is_fact_query`.
# Lý do: Router theo DIEM/HOU đã đủ. Ta dùng (1) scope và (2) có-năm hay không để chọn chiến lược retrieve.


def year_priority_filter(query: str, docs: List[Document]) -> List[Document]:
    match = YEAR_RE.search(query)
    if not match:
        return docs
    target_year = match.group(1)

    def document_has_year(d: Document) -> bool:
        meta = d.metadata or {}
        combined_text = (d.page_content or "") + " " + " ".join(
            str(meta.get(k, "")) for k in ["date", "title", "url", "section", "category", "year"]
        )
        return target_year in combined_text

    docs_with_year = [d for d in docs if document_has_year(d)]
    docs_without_year = [d for d in docs if not document_has_year(d)]
    return docs_with_year + docs_without_year


# [SỬA] Khử trùng lặp khi gộp kết quả từ nhiều kho
def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen_keys = set()
    unique_documents: List[Document] = []
    for doc in documents:
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", doc.metadata.get("loc", "")),
            hash(doc.page_content),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_documents.append(doc)
    return unique_documents


# [SỬA] Quyết định phạm vi tìm kiếm dựa trên câu hỏi và section
def decide_scope(query: str, section_value: str) -> str:
    query_lower = query.lower()
    if any(k in query_lower for k in DIEM_KEYWORDS):
        return "diem"
    if (any(k in query_lower for k in HOU_KEYWORDS)) or (section_value in
        ["giới thiệu", "tuyển sinh", "ngành học", "sự kiện", "thông báo", "hợp tác", "học phí"]):
        return "hou"
    return "both"


def retrieve_with_threshold(query: str, min_score: float = 0.35, k_cap: int = 100) -> List[Document]:
    # [SỬA] Duyệt trên "selected_vector_stores" (HOU/Điểm/cả hai) thay vì chỉ HOU
    scored_documents: List[tuple[Document, float]] = []

    for store in selected_vector_stores:
        results = store.similarity_search_with_relevance_scores(query, k=k_cap)
        for doc, score in results:
            numeric_score = score if score is not None else 0.0
            if (score is None) or (numeric_score >= min_score):
                scored_documents.append((doc, numeric_score))

    # Nếu không có kết quả vượt ngưỡng, nới lỏng: lấy top-k thô
    if not scored_documents:
        for store in selected_vector_stores:
            results = store.similarity_search_with_relevance_scores(query, k=k_cap)
            for doc, score in results:
                scored_documents.append((doc, score if score is not None else 0.0))

    scored_documents.sort(key=lambda item: item[1], reverse=True)
    merged_documents = deduplicate_documents([doc for doc, _ in scored_documents])

    # Nếu sau lọc còn quá ít → trả thêm một ít top (tối đa 10)
    if len(merged_documents) < 4:
        raw_documents: List[Document] = []
        for store in selected_vector_stores:
            raw_documents.extend([doc for doc, _ in store.similarity_search_with_relevance_scores(query, k=k_cap)])
        merged_documents = deduplicate_documents(raw_documents)[:10]

    return merged_documents


def retrieve_with_mmr(query: str, section_value: str) -> List[Document]:
    # [SỬA] Chạy MMR trên các kho đã được định tuyến
    # Kho HOU có 'section' ổn định → áp filter theo section để tăng phù hợp
    collected_documents: List[Document] = []

    for store in selected_vector_stores:
        try:
            if store is vectordb_hou:
                docs = store.max_marginal_relevance_search(
                    query=query,
                    k=100,
                    fetch_k=100,
                    lambda_mult=0.5,
                    filter=lambda meta: section_value in meta.get("section", ""),
                )
            else:
                docs = store.max_marginal_relevance_search(
                    query=query,
                    k=100,
                    fetch_k=100,
                    lambda_mult=0.5,
                )
        except Exception:
            docs = []
        collected_documents.extend(docs)

    unique_documents = deduplicate_documents(collected_documents)

    # Fallback MMR không filter nếu vẫn thiếu tài liệu
    if not unique_documents:
        fallback_documents: List[Document] = []
        for store in selected_vector_stores:
            try:
                fallback_documents.extend(
                    store.max_marginal_relevance_search(query=query, k=100, fetch_k=100, lambda_mult=0.5)
                )
            except Exception:
                pass
        unique_documents = deduplicate_documents(fallback_documents)

    return unique_documents

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

    # [SỬA] Định tuyến trước: chỉ Điểm / chỉ HOU / cả hai
    scope = decide_scope(query, section_value)
    global selected_vector_stores
    if scope == "diem":
        selected_vector_stores = [vectordb_diem]
    elif scope == "hou":
        selected_vector_stores = [vectordb_hou]
    else:
        selected_vector_stores = [vectordb_hou, vectordb_diem]
    print(f"🔀 Router scope: {scope} (số kho: {len(selected_vector_stores)})")

    # [SỬA] BỎ `is_fact_query`: thay bằng chiến lược dựa trên scope và năm
    # - Nếu scope == 'diem' hoặc câu hỏi có năm → threshold trước, thiếu thì MMR
    # - Ngược lại → MMR trước, thiếu thì threshold
    has_year = bool(YEAR_RE.search(query))
    if scope == "diem" or has_year:
        documents = retrieve_with_threshold(query, min_score=0.35, k_cap=100)
        if len(documents) < 4:
            documents = retrieve_with_mmr(query, section_value)
    else:
        documents = retrieve_with_mmr(query, section_value)
        if not documents:
            documents = retrieve_with_threshold(query, min_score=0.35, k_cap=100)

    # Ưu tiên theo năm (nếu có năm trong câu hỏi)
    documents = year_priority_filter(query, documents)
    return {**state, "documents": documents}

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
        # Truyền cả lịch sử hội thoại để hỗ trợ tham chiếu ngữ cảnh khi cần
        resp = answer_chain.invoke({"context": docs_text, "messages": state["messages"], "question": state["query"]})
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
        result = chatbot.invoke({"query": q}, cfg)
        print("\n📌 Trả lời:", result.get("answer"))
