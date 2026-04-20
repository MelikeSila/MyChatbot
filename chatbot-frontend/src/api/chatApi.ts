export const sendMessage = async (message: string) => {
  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:5001";
  const res = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  console.log("MESSAGES:", res);
  return res.json();
};