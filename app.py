from flask import Flask, request, jsonify
from flask_cors import CORS # Thư viện để xử lý CORS
import requests
import json
import time
import os # Để đọc biến môi trường

app = Flask(__name__)
# Cho phép CORS từ mọi nguồn. Trong môi trường production, bạn nên chỉ định rõ nguồn gốc cho phép.
CORS(app)

# Lấy API Key từ biến môi trường. Đây là cách tốt nhất để giữ an toàn cho khóa API của bạn.
# Nếu bạn chạy cục bộ mà không đặt biến môi trường, hãy thay thế os.getenv(...) bằng khóa API của bạn.
# Ví dụ: GEMINI_API_KEY = "YOUR_ACTUAL_GEMINI_API_KEY"
GEMINI_API_KEY = "AIzaSyC-AUp7NplKO6Y1RRtjwdSu6tRe2aqsknU"
if not GEMINI_API_KEY:
    print("Cảnh báo: Không tìm thấy GEMINI_API_KEY trong biến môi trường. "
          "Vui lòng đặt biến môi trường 'GEMINI_API_KEY' hoặc điền trực tiếp vào mã.")
    # Fallback cho Canvas runtime hoặc nếu bạn muốn đặt trực tiếp khi phát triển
    # GEMINI_API_KEY = "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY_NẾU_CHẠY_CỤC_BỘ_MÀ_KHÔNG_DÙNG_BIẾN_MÔI_TRƯỜNG"


def call_gemini_api(prompt, retries=3, delay=1):
    """
    Calls the Gemini API to get a response for the given prompt,
    with exponential backoff for retries.
    """
    chat_history = []
    chat_history.append({"role": "user", "parts": [{"text": prompt}]})
    payload = {"contents": chat_history}

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"

    for i in range(retries):
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, json=payload)
            response.raise_for_status()  # Ném lỗi cho các mã trạng thái HTTP xấu (4xx hoặc 5xx)

            result = response.json()
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                return result["candidates"][0]["content"]["parts"][0].get("text", "Không có nội dung phản hồi.")
            else:
                return "Cấu trúc phản hồi API không mong muốn hoặc thiếu nội dung."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and i < retries - 1:
                print(f"Lỗi giới hạn tốc độ (429). Đang thử lại sau {delay} giây...")
                time.sleep(delay)
                delay *= 2  # Tăng gấp đôi độ trễ
            else:
                print(f"Đã xảy ra lỗi HTTP: {e}")
                return f"Đã xảy ra lỗi HTTP: {e}"
        except requests.exceptions.ConnectionError as e:
            if i < retries - 1:
                print(f"Lỗi kết nối. Đang thử lại sau {delay} giây...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Lỗi kết nối: {e}")
                return f"Lỗi kết nối: {e}"
        except Exception as e:
            print(f"Đã xảy ra lỗi không mong muốn: {e}")
            return f"Đã xảy ra lỗi không mong muốn: {e}"
    return "Không thể kết nối với API Gemini sau nhiều lần thử lại."


# ✅ Trang chủ - Giao diện HTML
from flask import render_template
@app.route('/')
def home():
    return render_template("index.html")  # Yêu cầu file templates/index.html


@app.route('/ask_gemini', methods=['POST'])
def ask_gemini():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Vui lòng cung cấp câu hỏi"}), 400

    answer = call_gemini_api(question)
    if answer.startswith("Lỗi"):
        return jsonify({"error": answer}), 500
    return jsonify({"answer": answer})

# ✅ Chạy Flask
if __name__ == '__main__':
    app.run(debug=True, port=5000)
