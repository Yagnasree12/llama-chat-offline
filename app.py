# Make sure Streamlit is installed before running
try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("Streamlit is not installed. Run `pip install streamlit` before running this app.")

# Then continue with the original content

"""
**LLaMate — Your Private Offline AI Assistant**
An ultra‑polished Streamlit Web UI for chatting with local LLaMA models served by **Ollama**.
-------------------------------------------------------------------------------
Run it locally:
```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\\Scripts\\activate)
pip install -r requirements.txt
streamlit run app.py
```

Requirements (`requirements.txt`):
```
streamlit>=1.34
requests
pillow
speechrecognition
```

Features 🧪
- Gradient background & glass‑morphism chat bubbles 🎨
- Emoji‑friendly messages 😄
- **TXT** document upload → used as context
- **Image** upload for visual Q&A 🖼️ (image encoded + prompt)
- Custom avatars (🧑 / 🤖)
- Voice input via microphone (offline with PocketSphinx if installed, else Google API) 🎙️
- Markdown rendering in replies 📘
- Sidebar accent color picker, temperature & top‑p sliders
- Light ⇄ dark theme switch
- Scroll‑locked chat area, auto‑scroll to newest
- One‑click download of transcript (TXT / JSON)
- Ollama health badge

Built entirely client‑side—no data leaves your machine.
"""

# ...[rest of code remains unchanged, as previously saved]...
