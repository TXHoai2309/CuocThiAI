# chatbot/query_chat.py
import os
import re
from typing import TypedDict, List
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
     "Ưu tiên chính xác, logic, dễ hiểu, và chỉ dùng thông tin trong phần 'Tài liệu'."),
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
    docs_text = truncate_docs(state["documents"], max_chars=4000)  # Giữ ngưỡng 4k ký tự để giảm rủi ro tràn token
    if not docs_text.strip():
        print("⚠️ Empty context detected!")
        return {**state, "answer": "Không tìm thấy thông tin liên quan."}

    # CHANGED: gọi answer_chain.invoke với {context, question}, thay vì format PromptTemplate rồi llm.invoke chuỗi.
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
