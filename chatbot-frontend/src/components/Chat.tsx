import { useState } from "react";
import { sendMessage } from "../api/chatApi";

type Message = {
  role: "user" | "bot";
  text: string;
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input) return;

    const userMessage: Message = { role: "user", text: input };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendMessage(input);

      const botMessage: Message = {
        role: "bot",
        text: data.reply,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: "40px auto" }}>
      <h2>AI Chatbot</h2>

      <div
        style={{
          border: "1px solid #ccc",
          height: 400,
          overflowY: "auto",
          padding: 10,
        }}
      >
        {messages.map((msg, i) => (
          <div key={i}>
            <b>{msg.role === "user" ? "You" : "Bot"}:</b> {msg.text}
          </div>
        ))}

        {loading && <p>Typing...</p>}
      </div>

      <div style={{ display: "flex", marginTop: 10 }}>
        <input
          style={{ flex: 1 }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
}