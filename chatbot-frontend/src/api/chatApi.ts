export const sendMessage = async (message: string) => {
  const API_URL = import.meta.env.VITE_API_URL;
  const res = await fetch(API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  if (!res.ok) {
    throw new Error("API error");
  }

  return res.json();
};