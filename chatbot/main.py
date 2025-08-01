# app.py
from flask import Flask, request, jsonify, send_from_directory
from query_chat import chatbot           # import graph đã compile sẵn
import threading, webbrowser

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    # chat.html nằm cùng thư mục với app.py
    return send_from_directory('.', 'chat.html')

@app.post('/chat')
def chat():
    data = request.get_json() or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({"answer": "Vui lòng nhập câu hỏi."}), 400
    result = chatbot.invoke({"query": query})
    return jsonify({"answer": result["answer"]})

if __name__ == '__main__':
    url = "http://127.0.0.1:5000/"
    print(f"➡️  Mở trình duyệt tại: {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(debug=True)
