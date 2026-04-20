from flask import Flask, request, jsonify
from flask_cors import CORS
import ChatbotTextToText
import os

app = Flask(__name__)
chatbot = ChatbotTextToText.ChatbotTextToText()

# Frontend (React) erişimi için kritik
CORS(app)

@app.route("/")
def home():
    return "Backend is running"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_message = data.get("message", "")

    bot_reply = generate_response(user_message)
    return jsonify({
        "reply": bot_reply
    })


def generate_response(message: str) -> str:
    message = message.lower()

    answer = chatbot.ask_to_chatbot(message)

    return  answer

if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5173)
    port = int(os.environ.get("PORT", 5173))
    app.run(host="0.0.0.0", port=port)