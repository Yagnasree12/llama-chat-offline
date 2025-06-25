# llama-chat-offline--LLaMATE
💬 Offline Chatbot UI using Streamlit + Ollama + LLaMA + Image/Voice/Doc Q&amp;A

An ultra-modern, **fully offline chatbot** UI using [Ollama](https://ollama.com), [LLaMA 3](https://ollama.com/library/llama3), and [Streamlit](https://streamlit.io).

## 🚀 Features

🧠 LLaMA-3 (8B) via Ollama – fast local inference, open-weights

💬 Real-time Chat UI – markdown, emoji, avatars, clean layout

📄 PDF / TXT Upload – inject document context into the chat

🖼️ Image Upload – ask questions based on images (optional)

🔒 Private & Secure – nothing leaves your device

⚙️ Adjustable AI settings – temperature, top‑p, model selection

🧹 One-click Clear Chat / Context

🎨 Custom dark UI with Streamlit + CSS

🚫 No API keys or cloud needed

🛠️ Built With
Python

Streamlit

Ollama

LLaMA

pypdf

Pillow


---

## 📦 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/llama-chat-ui-offline.git
cd llama-chat-ui-offline

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\\Scripts\\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama and run model
ollama run llama3

# 5. Launch the app
streamlit run app.py

