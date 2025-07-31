import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, START
from langchain.chains import LLMChain

# === API Key ===
os.environ["GOOGLE_API_KEY"] = "AIzaSyDR1eVkKtTN3RBeXNdW3bThRIwMMMfJND8"

# === Load FAISS ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(
    r"D:\airdrop\CuocThiAI\hou_crawler\crawler\data\hou_index",
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True
)

# === Gemini model ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.5,
    max_output_tokens=512
)

# === Prompt trả lời chính ===
main_prompt = PromptTemplate.from_template("""Bạn là một trợ lý AI thông minh, trả lời câu hỏi dựa trên dữ liệu đã được tinh chỉnh. Hãy cung cấp câu trả lời chính xác, ngắn gọn và dễ hiểu dựa trên thông tin từ tài liệu sau:

**Tài liệu**: {context}

**Câu hỏi**: {question}

**Trả lời**:""")

# === Prompt phân loại section ===
section_prompt = PromptTemplate.from_template(
    "Phân tích câu hỏi sau và trả lời bằng đúng 1 từ trong: 'tuyển sinh', 'ngành học', 'sự kiện', 'thông báo' , 'hợp tác' , 'học phí' , 'khác'.\n\nCâu hỏi: {query}\nPhân loại:"
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
    if section not in ["tuyển sinh", "ngành học", "sự kiện", "thông báo", "hợp tác", "học phí", "khác"]:
        section = "khác"
    print(f"📎 Gemini xác định section: {section}")
    return {**state, "section": section}


# === Node 2: Truy xuất tài liệu từ FAISS ===
def retrieve_docs(state: QAState) -> QAState:
    docs = vectordb.similarity_search(
        query=state["query"],
        k=50,
        filter=lambda meta: state["section"] in meta.get("section", "")
    )
    return {**state, "documents": docs}


# === Node 3: Tạo câu trả lời từ context ===
def generate_answer(state: QAState) -> QAState:
    docs_text = "\n\n".join(doc.page_content for doc in state["documents"])
    prompt = main_prompt.format(context=docs_text, question=state["query"])
    output = llm.invoke(prompt).content
    return {**state, "answer": output}


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
