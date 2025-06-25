# llama-chat-offline--LLaMATE
💬 Offline Chatbot UI using Streamlit + Ollama + LLaMA + Image/Voice/Doc Q&amp;A

An ultra-modern, **fully offline chatbot** UI using [Ollama](https://ollama.com), [LLaMA 3](https://ollama.com/library/llama3), and [Streamlit](https://streamlit.io).

## 🚀 Features

✅ Chat with powerful open-source LLMs  
✅ 100% Private – Works fully offline (No internet needed)  
✅ Upload documents (TXT) and ask questions  
✅ Upload images for Visual Q&A 🖼️  
✅ Voice input support 🎤  
✅ Customizable UI with avatars, markdown, and themes  
✅ Download chat as `.txt` or `.json`  
✅ Emoji & markdown-friendly formatting  
✅ Responsive and user-friendly design

🛠️ Built With
Python

Streamlit

Ollama

LLaMA

SpeechRecognition

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

