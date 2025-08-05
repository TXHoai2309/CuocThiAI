# chatbot/query_chat.py
import os
import re
import unicodedata  # [NEW] dùng để chuẩn hóa bỏ dấu tiếng Việt
from typing import Annotated, TypedDict, List, Sequence

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# === API Key ===
os.environ.setdefault("GOOGLE_API_KEY", "AIzaSyDR1eVkKtTN3RBeXNdW3bThRIwMMMfJND8")

# === Load FAISS ===

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
base_dir = os.path.dirname(os.path.abspath(__file__))

embedding_model_ctdt = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Đường dẫn đến index FAISS
hou_index_path = os.path.join(base_dir, "..", "data", "hou", "hou_index")
diem_index_path = os.path.join(base_dir, "..", "data", "hou", "diem_chuan_index")

ctdt_index_path = os.path.join(base_dir, "..", "data", "hou", "ctdt_index")
if not os.path.exists(ctdt_index_path):  # [NEW] fallback path
    ctdt_index_path = os.path.join(base_dir, "..", "data", "ctdt_index")

# Tải index FAISS với embeddings
vectordb_hou = FAISS.load_local(
    hou_index_path,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True,
)

vectordb_diem = FAISS.load_local(
    diem_index_path,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True,
)

vectordb_ctdt = FAISS.load_local(
    ctdt_index_path,
    embeddings=embedding_model_ctdt,
    index_name="index",  # mặc định của save_local
    allow_dangerous_deserialization=True,
)

# [CHANGE] Danh sách kho sẽ được router cập nhật theo truy vấn
selected_vector_stores: List[FAISS] = [vectordb_hou]

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
        "Ưu tiên chính xác, logic, dễ hiểu, và chỉ dùng thông tin trong phần 'Tài liệu'. "
        "Với ngữ cảnh hội thoại (ví dụ: tên người dùng, tham chiếu như 'ngành này'), bạn được phép dùng lịch sử hội thoại. "
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

YEAR_RE = re.compile(r"(20\d{2})")


# [NEW] Chuẩn hóa tiếng Việt: lowercase + bỏ dấu + thay '_' thành ' ' + gọn khoảng trắng
def vn_normalize(s: str) -> str:
    s = (s or "").lower().strip().replace("_", " ")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s)
    return s

# [NEW] Bộ keyword mở rộng (không dấu) cho 3 kho và bộ phủ định để giảm nhầm lẫn
CTDT_KEYWORDS_RAW = [
    # Thuật ngữ chung về chương trình đào tạo
    "ctđt", "chương trình đào tạo", "khung chương trình", "kế hoạch đào tạo", "kế hoạch học tập",
    "chương trình học", "khối kiến thức", "khối_kiến_thức", "khối ngành", "chuyên ngành", "ngành học",
    # Thành phần và học phần
    "học phần", "môn học", "môn", "mô tả học phần", "học phần bắt buộc", "học phần tự chọn",
    "học phần tự chọn tự do", "số tín chỉ", "tín chỉ", "tin chi", "thời lượng", "học kỳ",
    # Chuẩn đầu ra
    "clo", "plo", "chuẩn đầu ra", "chuẩn đầu ra học phần", "chuẩn đầu ra chương trình", "mục tiêu đào tạo",
    # Thông tin bổ sung
    "thời gian đào tạo", "thời_gian_đào_tạo", "vị trí việc làm", "vị_trí_việc_làm",
    "cơ hội nghề nghiệp", "yêu cầu đầu vào", "điều kiện tiên quyết",
]
DIEM_KEYWORDS_RAW = [
    "điểm", "điểm chuẩn", "điểm sàn", "ngưỡng", "điểm xét tuyển", "chỉ tiêu", "mã ngành", "tổ hợp",
    "điểm thi", "điểm học bạ", "điểm đánh giá năng lực", "điểm tư duy", "xét tuyển", "thpt", "học bạ", "đgnl", "đgtd",
    "a00", "a01", "b00", "d01", "d07",
]
# [FIX] Thiếu dấu phẩy giữa "tuyển sinh" và "chuyên ngành" -> gây dính chuỗi
HOU_KEYWORDS_RAW = [
    "học phí", "thông báo", "tin tức", "sự kiện", "tuyển sinh",
    "giới thiệu", "hợp tác", "đối tác",
    "khoa", "viện", "phòng ban", "bộ môn", "giảng viên", "cán bộ", "hồ sơ", "thủ tục",
    "quy chế", "lịch học", "lịch thi", "lịch nghỉ", "hoạt động ngoại khóa", "câu lạc bộ",
]


NEGATE_FOR_CTDT = ["diem chuan", "diem san", "to hop", "ma nganh"]
NEGATE_FOR_DIEM = ["tin chi", "hoc phan", "syllabus", "clo", "plo"]

CTDT_KEYWORDS = {vn_normalize(x) for x in CTDT_KEYWORDS_RAW}
DIEM_KEYWORDS = {vn_normalize(x) for x in DIEM_KEYWORDS_RAW}
HOU_KEYWORDS = {vn_normalize(x) for x in HOU_KEYWORDS_RAW}
NEGATE_FOR_CTDT = {vn_normalize(x) for x in NEGATE_FOR_CTDT}
NEGATE_FOR_DIEM = {vn_normalize(x) for x in NEGATE_FOR_DIEM}


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

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen_keys = set()
    unique_documents: List[Document] = []
    for doc in documents:
        key = (
            (doc.metadata or {}).get("source", ""),
            (doc.metadata or {}).get("page", (doc.metadata or {}).get("loc", "")),
            hash(doc.page_content),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_documents.append(doc)
    return unique_documents

# [CHANGE] Router: ưu tiên Điểm → CTĐT → HOU; dùng chuỗi đã chuẩn hóa không dấu
def decide_scope(query: str, section_value: str) -> str:
    q = vn_normalize(query)

    if any(k in q for k in DIEM_KEYWORDS) and not any(k in q for k in NEGATE_FOR_DIEM):
        return "diem"

    if any(k in q for k in CTDT_KEYWORDS) and not any(k in q for k in NEGATE_FOR_CTDT):
        return "ctdt"

    if (any(k in q for k in HOU_KEYWORDS)) or (section_value in
        ["giới thiệu", "tuyển sinh", "ngành học", "sự kiện", "thông báo", "hợp tác", "học phí"]):
        return "hou"

    return "both"

def retrieve_with_threshold(query: str, min_score: float = 0.35, k_cap: int = 100) -> List[Document]:
    scored_documents: List[tuple[Document, float]] = []

    for store in selected_vector_stores:
        results = store.similarity_search_with_relevance_scores(query, k=k_cap)
        for doc, score in results:
            numeric_score = score if score is not None else 0.0
            if (score is None) or (numeric_score >= min_score):
                scored_documents.append((doc, numeric_score))

    if not scored_documents:
        for store in selected_vector_stores:
            results = store.similarity_search_with_relevance_scores(query, k=k_cap)
            for doc, score in results:
                scored_documents.append((doc, score if score is not None else 0.0))

    scored_documents.sort(key=lambda item: item[1], reverse=True)
    merged_documents = deduplicate_documents([doc for doc, _ in scored_documents])

    if len(merged_documents) < 4:
        raw_documents: List[Document] = []
        for store in selected_vector_stores:
            raw_documents.extend([doc for doc, _ in store.similarity_search_with_relevance_scores(query, k=k_cap)])
        merged_documents = deduplicate_documents(raw_documents)[:10]

    return merged_documents

def retrieve_with_mmr(query: str, section_value: str) -> List[Document]:
    collected_documents: List[Document] = []

    for store in selected_vector_stores:
        try:
            docs = store.max_marginal_relevance_search(
                query=query,
                k=100,
                fetch_k=100,
                lambda_mult=0.5,
            )
        except Exception:
            docs = []
        collected_documents.extend(docs)

    # [NEW] Lọc theo section sau khi lấy MMR (áp dụng tốt với HOU/CTĐT)
    if section_value and section_value != "khác":
        def match_section(d: Document) -> bool:
            sec = (d.metadata or {}).get("section", "")
            return section_value in sec
        post = [d for d in collected_documents if match_section(d)]
        if post:
            collected_documents = post

    unique_documents = deduplicate_documents(collected_documents)

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

    scope = decide_scope(query, section_value)
    global selected_vector_stores
    if scope == "diem":
        selected_vector_stores = [vectordb_diem]
    elif scope == "ctdt":
        selected_vector_stores = [vectordb_ctdt]
    elif scope == "hou":
        selected_vector_stores = [vectordb_hou]
    else:
        # [CHANGE] BOTH: ưu tiên HOU + CTĐT rồi mới đến Điểm
        selected_vector_stores = [vectordb_hou, vectordb_ctdt, vectordb_diem]
    print(f"🔀 Router scope: {scope} (số kho: {len(selected_vector_stores)})")

    has_year = bool(YEAR_RE.search(query))
    if scope in ("diem",) or has_year:
        documents = retrieve_with_threshold(query, min_score=0.35, k_cap=100)
        if len(documents) < 4:
            documents = retrieve_with_mmr(query, section_value)
    else:
        documents = retrieve_with_mmr(query, section_value)
        if not documents:
            documents = retrieve_with_threshold(query, min_score=0.35, k_cap=100)

    documents = year_priority_filter(query, documents)
    return {**state, "documents": documents}

# === Node 3: trả lời + đẩy AIMessage vào bộ nhớ ===
def truncate_docs(documents: List[Document], max_chars=4000) -> str:
    combined = ""
    for doc in documents:
        header = []
        meta = doc.metadata or {}
        # [NEW] Nhúng thêm meta vào ngữ cảnh để LLM trả lời chính xác hơn
        for k in ("section", "major", "course_name", "course_code", "credits", "semester", "year"):
            if meta.get(k) not in (None, ""):
                header.append(f"{k}: {meta.get(k)}")
        header_text = ("[" + " | ".join(header) + "]\n") if header else ""
        block = header_text + (doc.page_content or "")
        if len(combined) + len(block) < max_chars:
            combined += block + "\n\n"
        else:
            break
    return combined

def generate_answer(state: State) -> State:
    docs_text = truncate_docs(state["documents"], max_chars=4000)
    if not docs_text.strip():
        content = "Không tìm thấy thông tin liên quan."
        return {**state, "messages": [AIMessage(content=content)], "answer": content}

    try:
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
    # In ra path đang dùng để dễ debug
    print("[INFO] HOU index:", hou_index_path)
    print("[INFO] DIEM index:", diem_index_path)
    print("[INFO] CTDT index:", ctdt_index_path)
    while True:
        q = input("❓ Hỏi: ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            break
        result = chatbot.invoke({"query": q}, cfg)
        print("\n📌 Trả lời:", result.get("answer"))
