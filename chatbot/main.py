from flask import Flask, request, jsonify, send_from_directory
from query_chat import chatbot
from langchain_core.messages import HumanMessage
import threading, webbrowser

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'chat.html')

@app.post('/chat')
def chat():
    data = request.get_json() or {}
    query = (data.get('query') or '').strip()
    thread_id = (data.get('thread_id') or 'default').strip()
    if not query:
        return jsonify({"answer": "Vui lòng nhập câu hỏi."}), 400

    config = {"configurable": {"thread_id": thread_id} or "default"}
    # Gọi graph với messages để bộ nhớ được nối tự động
    result = chatbot.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    return jsonify({"answer": result.get("answer", "")})

if __name__ == '__main__':
    url = "http://127.0.0.1:5000/"
    print(f"➡️  Mở trình duyệt tại: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=True)