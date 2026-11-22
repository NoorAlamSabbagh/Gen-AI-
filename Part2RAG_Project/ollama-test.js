
const testOllama = async () => {
  try {
    const response = await fetch(
      "http://localhost:11434/api/chat",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: "qwen3:4b",
          messages: [
            {
              role: "user",
              content: "Explain RAG in simple terms."
            }
          ],
          stream: false
        })
      }
    );

    const data = await response.json();

    console.log(data.message.content);
  } catch (error) {
    console.error("Ollama error:", error.message);
  }
};

testOllama();
