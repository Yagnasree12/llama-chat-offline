# ===============================================================
# LLaMate — Your Private Offline AI Assistant (Optimised, Fast, No Voice)
# ===============================================================
"""
💬 **LLaMate** is an offline Streamlit UI that chats with **LLaMA‑3** (via Ollama)
with **image and document upload**, emojis, markdown, avatars — no internet needed.

### Quick start (Windows example)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama run llama3        # keep this terminal running (or 'ollama serve' if you have multiple models)
# new terminal
streamlit run app.py      # ALWAYS launch with streamlit
requirements.txt

streamlit>=1.34
requests
pillow
pypdf               # Added for PDF support
Key Features & Enhancements
🧠 Offline Local Chat (Ollama + LLaMA-3 for fast, private conversations)
🖼️ Image & 📝 Text Uploads (for contextual understanding)
Supports PDF documents for summarization and context.
🗨️ Markdown + Emojis + Avatars (rich, expressive chat)
⚡ Lightning-fast & Efficient:
Leverages Ollama's streaming API and Streamlit's st.write_stream for immediate, real-time token display as the AI generates, giving a highly responsive feel.
Employs Streamlit's caching (st.cache_data) for efficient handling of file operations (reading documents, resizing images), avoiding redundant processing.
Minimal client-side processing to ensure responsiveness.
Note on speed: Actual token generation speed (AI's "thinking" time) depends heavily on your local hardware (CPU, GPU, RAM) and the size of the LLaMA-3 model chosen. This application ensures the fastest possible display of results your system can produce.
🚫 No Internet or API keys needed (truly private)
⚙️ User-Friendly & Interactive:
Clean, intuitive layout with clear navigation.
Dedicated "Clear Chat", "Clear Text Context", and "Clear Image" buttons for easily managing conversation and context.
Clear status indicators for Ollama server connectivity.
Adjustable AI parameters (Temperature, Top-p) for custom response generation.
🎨 Attractive & Unique UI:
Custom CSS provides a modern, distinct aesthetic for chat messages, sidebar, and controls.
Responsive design adapts well to various screen sizes (mobile-friendly).
✅ Error-Free & Robust:
Comprehensive error handling for Ollama communication (connection issues, timeouts, HTTP errors).
Specific fixes for StreamlitValueAssignmentNotAllowedError and Cannot hash argument caching issues.
Guards against common operational warnings and removes redundant st.rerun() calls within on_submit callbacks. """
# ---------------------------------------------------------------
# Imports & checks
# ---------------------------------------------------------------
try:
    import streamlit as st
    import requests
    from PIL import Image
    import base64
    import io
    import json
    import pypdf  # New import for PDF processing
except ModuleNotFoundError as exc:
    st.error(
        f"Error: Missing required library: {exc.name}. "
        "Please ensure all dependencies are installed by running "
        "pip install -r requirements.txt."
    )
    st.stop()
except ImportError as exc:
    st.error(f"Error: Failed to import a library: {exc}. "
             "This might indicate a corrupted installation or a path issue. "
             "Please try reinstalling your requirements.")
    st.stop()

# ---------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_file_as_text(f):
    """Reads an uploaded text file as text, handling potential decoding issues."""

    try:
        f.seek(0)
        return f.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error reading text file: {e}. Please ensure it's a valid TXT file.")
        return ""


@st.cache_data(show_spinner=False)
def read_pdf_as_text(f):
    """Extracts text from an uploaded PDF file."""

    try:
        f.seek(0)
        reader = pypdf.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle pages with no extractable text
        return text
    except pypdf.errors.PdfReadError:
        st.error("Error reading PDF: The file might be corrupted or encrypted. Cannot extract text. Try a different PDF.")
        return ""
    except Exception as e:
        st.error(f"An unexpected error occurred reading PDF: {e}")
        return ""


@st.cache_data(show_spinner=False)
def resize_image(_img: Image.Image, max_px=512):
    """Resizes an image while maintaining aspect ratio for efficient LLM processing."""

    img_copy = _img.copy()
    img_copy.thumbnail((max_px, max_px))
    return img_copy


@st.cache_data(show_spinner=False)
def img_to_b64(img: Image.Image) -> str:
    """Converts a PIL Image to a base64 encoded string."""

    buf = io.BytesIO()
    if img.mode not in ['RGB', 'RGBA']:
        img = img.convert('RGB')
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def check_ollama_status():
    """Checks if the Ollama server is running and accessible."""

    try:
        requests.get("http://localhost:11434/", timeout=3)
        return True
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False

# ---------------------------------------------------------------
# Streamlit session defaults
# ---------------------------------------------------------------


def init_state():
    """Initializes Streamlit session state variables with default values."""

    defaults = dict(
        history=[],
        model="llama3",
        ctx_txt="",
        img_b64="",
        temp=0.6,
        top_p=0.9,
        accent="#0099ff",
        ollama_reachable=False,
        chat_disabled=False
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

# ---------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------


def sidebar():
    """Renders the sidebar for context upload and model settings."""

    with st.sidebar:
        st.header("📁 Context Upload")
        # Shortened text as requested
        st.info(
            "Upload TXT/PDF documents for context or summarization. "
            "Upload images for visual AI understanding."
        )

        doc = st.file_uploader(
            "TXT or PDF Document (≤5 KB recommended)",
            type=["txt", "pdf"],  # Now accepts PDF
            label_visibility="collapsed",
            key="doc_uploader"
        )
        if doc:
            with st.spinner("Processing document..."):
                file_content = ""
                if doc.type == "text/plain":
                    file_content = read_file_as_text(doc)
                elif doc.type == "application/pdf":
                    file_content = read_pdf_as_text(doc)
                else:
                    st.warning("Unsupported file type uploaded.")

                if file_content:
                    # Limit context to 5000 characters
                    original_len = len(file_content)
                    st.session_state.ctx_txt = file_content[:5000]

                    if original_len > 5000:
                        st.warning(
                            f"Document content truncated to 5000 characters. Original length: {original_len} characters.")
                        # Enhanced warning for PDF processing time
                        st.warning(
                            "Processing large PDFs can take a significant amount of time, depending on your system's performance.")
                    st.success("Document context loaded! You can now ask the AI to summarize it or discuss its content.")
                else:
                    st.warning(
                        "Could not read content from the uploaded document. It might be empty, corrupted, or an unsupported format.")

        # Clear text context button: Moved outside `if doc` block so it's always available
        if st.session_state.ctx_txt and st.button("Clear Text Context", key="clear_text_ctx"):
            st.session_state.ctx_txt = ""
            st.info("Text context cleared.")
            st.rerun()  # This rerun is necessary for explicit button actions

        img = st.file_uploader(
            "Image (optional)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="img_uploader"
        )
        if img:
            with st.spinner("Processing image..."):
                try:
                    img.seek(0)
                    im = Image.open(img)
                    im_resized = resize_image(im)
                    st.session_state.img_b64 = img_to_b64(im_resized)
                    st.image(im_resized, caption="Uploaded Image", use_column_width=True)
                    st.success("Image loaded! (Resized for efficiency)")
                except Exception as e:
                    st.error(f"**Error processing image:** {e}. Please ensure it's a valid image file (PNG, JPG, JPEG).")
                    st.session_state.img_b64 = ""

        # Clear image context button: Moved outside `if img` block so it's always available
        if st.session_state.img_b64 and st.button("Clear Image", key="clear_img_ctx"):
            st.session_state.img_b64 = ""
            st.info("Image context cleared.")
            st.rerun()  # This rerun is necessary for explicit button actions

        st.divider()
        st.header("⚙️ Settings")

        st.session_state.model = st.selectbox(
            "Select Model",
            ["llama3"],
            index=0,
            help="Choose the LLM model to interact with. Ensure the selected model is installed via Ollama (`ollama pull <model_name>`)."
        )

        st.session_state.temp = st.slider(
            "Temperature", 0.0, 1.0, 0.6, step=0.05,
            help="Controls the randomness and creativity of the output. Higher values (e.g., 0.8) lead to more diverse and surprising responses, while lower values (e.g., 0.2) make the output more deterministic and focused."
        )
        st.session_state.top_p = st.slider(
            "Top‑p (Nucleus Sampling)", 0.0, 1.0, 0.9, step=0.05,
            help="Controls the diversity of the output by sampling from the most probable tokens. Lower values (e.g., 0.5) result in less diverse and more focused output, while higher values (e.95) allow for more varied responses."
        )

        st.divider()
        st.markdown("Developed with ❤️ by **Yagnasree**")  # Updated credit
# ---------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------


def ask_ollama(prompt: str):
    """
    Sends a request to the Ollama API and yields streamed responses for better interactivity.
    Includes robust error handling and conversational context management.
    """

    url = "http://localhost:11434/api/chat"

    messages = []
    # Add system context if available (from document upload)
    if st.session_state.ctx_txt:
        messages.append({"role": "system", "content": st.session_state.ctx_txt})

    # Include recent chat history for conversational context
    # Send a limited number of previous messages to manage token usage and improve speed
    num_history_messages = 6  # Total messages (user+assistant) to send as context

    # Filter out the very last user prompt if it's the one just entered,
    # as it will be added explicitly as the current user message.
    # This prevents sending duplicate user prompts in the history.
    effective_history = [msg for msg in st.session_state.history if
                         not (msg["role"] == "user" and msg["content"] == prompt)]
    messages.extend(effective_history[-num_history_messages:])

    # Add the current user prompt
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": st.session_state.model,
        "messages": messages,
        "stream": True,  # Critical for immediate, typing-like responses
        "temperature": st.session_state.temp,
        "top_p": st.session_state.top_p,
    }

    # Add image to payload if available (from image upload)
    if st.session_state.img_b64:
        payload["images"] = [st.session_state.img_b64]

    try:
        # Use requests.post with stream=True for chunked responses
        # Increased timeout to 180 seconds for potentially longer generations, especially with vision or complex prompts
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()  # Raise an exception for HTTP error codes (4xx or 5xx)
            full_response = ""
            for chunk in r.iter_content(chunk_size=None):  # Auto-determine chunk size
                if chunk:
                    try:
                        # Decode and parse each line as a JSON object
                        decoded_chunk = chunk.decode("utf-8")
                        for line in decoded_chunk.splitlines():
                            if line.strip():  # Ensure line is not empty
                                data = json.loads(line)
                                # Extract content from the message part of the response
                                if "message" in data and "content" in data["message"]:
                                    content = data["message"]["content"]
                                    full_response += content
                                    yield content  # Yield current content for streaming display
                                # Ollama indicates end of stream with "done": true
                                if data.get("done"):
                                    break  # Exit the loop if generation is complete
                    except json.JSONDecodeError:
                        # Log malformed JSON chunks but continue processing
                        print(f"Warning: Received malformed JSON chunk from Ollama: {line.strip()}")
                        continue  # Skip to the next chunk

            # If no content was received despite no error, provide a default message
            if not full_response.strip():
                yield "⚠️ No meaningful reply from Ollama. The model might not have generated any content for this prompt."

            return full_response  # Return the complete response string

    except requests.exceptions.ConnectionError:
        st.session_state.ollama_reachable = False
        error_message = "❌ **Ollama server not reachable.** Please ensure Ollama is running (`ollama serve`) and the selected model (`ollama run llama3`) is pulled and available in your terminal."
        yield error_message
        return error_message
    except requests.exceptions.Timeout:
        error_message = "⏳ **Ollama request timed out.** The model might be taking too long to respond. Try simplifying the prompt, reducing context, or adjusting timeout settings."
        yield error_message
        return error_message
    except requests.exceptions.HTTPError as e:
        # Catch specific HTTP errors (e.g., 404 for model not found)
        status_code = e.response.status_code
        error_message = f"❌ **Ollama HTTP error {status_code}:** {e.response.text.strip()}."
        if "model" in e.response.text and "not found" in e.response.text:
            error_message += " Please check if the selected model is correctly installed (`ollama pull <model_name>`)."
        yield error_message
        return error_message
    except requests.exceptions.RequestException as e:
        # Catch any other request-related errors
        error_message = f"❌ **An Ollama communication error occurred:** {e}"
        yield error_message
        return error_message
    except Exception as e:
        # Catch any other unexpected errors
        error_message = f"❌ An unexpected error occurred during AI interaction: {e}"
        yield error_message
        return error_message
# ---------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------


def chat_ui():
    """Renders the main chat interface, including message display and input."""

    # Use a container for the chat history with a fixed height and scrollbar
    # This prevents the chat history from pushing down the input box
    log = st.container(height=560, border=True)
    with log:
        for m in st.session_state.history:
            avatar = "🧑" if m["role"] == "user" else "🤖"
            # Using Streamlit's chat_message directly for consistent rendering and avatar support
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])  # Render markdown content

    # Auto-scroll to bottom using JavaScript for seamless user experience
    st.write("<script>window.scrollTo(0,document.body.scrollHeight);</script>", unsafe_allow_html=True)

    # Check Ollama status before enabling the chat input
    st.session_state.ollama_reachable = check_ollama_status()

    # Display warning and disable input if Ollama is not running
    if not st.session_state.ollama_reachable:
        st.warning(
            "**Ollama server is not running or accessible.** "
            "Please ensure Ollama is started in a separate terminal: "
            "`ollama serve` or `ollama run llama3`."
        )
        st.session_state.chat_disabled = True
    else:
        st.session_state.chat_disabled = False

    # Chat input area
    # The 'on_submit' callback is crucial for triggering generation upon input
    st.chat_input(
        "Ask me anything...",
        key="chat_input",  # Unique key for the input widget
        on_submit=process_chat_input,  # Function to call when input is submitted
        disabled=st.session_state.chat_disabled  # Disable if Ollama is not reachable
    )

    # Add a clear chat button for better user control
    if st.session_state.history:  # Only show clear button if there's history
        if st.button("Clear Chat", key="clear_chat_button", help="Start a new conversation"):
            st.session_state.history = []  # Clear the history
            st.session_state.ctx_txt = ""  # Also clear text context
            st.session_state.img_b64 = ""  # Also clear image context
            st.rerun()  # This rerun is necessary for explicit button actions

# Callback function for chat input submission
def process_chat_input():
    prompt = st.session_state.chat_input

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        response_generator = ask_ollama(prompt)
        full_answer = ""
        for chunk in st.write_stream(response_generator):
            full_answer += chunk

        if full_answer.strip():
            st.session_state.history.append({"role": "assistant", "content": full_answer})
        else:
            st.session_state.history.append({"role": "assistant", "content": "⚠️ No response generated by the model."})
    # No st.rerun() here; Streamlit implicitly reruns after on_submit completes
# ---------------------------------------------------------------
# Main application function
# ---------------------------------------------------------------


def main():
    """Main function to configure and run the Streamlit application."""

    st.set_page_config(
        page_title="LLaMate — Offline Chat with LLaMA",  # Browser tab title
        page_icon="🦙",  # Browser tab icon (llama emoji)
        layout="centered",  # Page layout: "centered" or "wide"
        initial_sidebar_state="expanded"  # Sidebar state on load
    )
    init_state()  # Initialize session state variables

    # Apply global custom CSS styling for enhanced aesthetics
    st.markdown(f"""
        <style>
            /* Define primary accent color variable for global use */
            :root {{
                --primary-color: {st.session_state.accent};
            }}

            /* General page background */
            body {{
                background-color: #f8f9fa; /* Light grey background */
            }}

            /* Main content block styling */
            .main .block-container {{
                padding-top: 2rem;
                padding-right: 1.5rem;
                padding-left: 1.5rem;
                padding-bottom: 2rem;
                max-width: 800px; /* Increased max-width for better use of space */
                margin-left: auto;
                margin-right: auto;
            }}

            /* Enhanced chat message container styling */
            .st-chat-message-container {{
                border-radius: 12px; /* More rounded corners */
                padding: 15px 20px; /* More padding for content */
                margin-bottom: 12px; /* Space between messages */
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); /* More pronounced shadow */
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; /* Smooth hover effects */
            }}
            .st-chat-message-container:hover {{
                transform: translateY(-3px); /* Noticeable lift on hover */
                box-shadow: 0 4px 10px rgba(0,0,0,0.15); /* Stronger shadow on hover */
            }}

            /* User message specific styles: pushed to right, distinct color */
            .st-chat-message-container[data-testid="stChatMessage"][data-author="user"] {{
                background-color: #e3f2fd; /* Light blue */
                border-top-right-radius: 4px; /* Slightly less rounded on the top-right to indicate direction */
                margin-left: 20%; /* Push user messages to the right */
                border: 1px solid #bbdefb; /* Subtle border */
            }}
            /* Assistant message specific styles: pushed to left, distinct color */
            .st-chat-message-container[data-testid="stChatMessage"][data-author="assistant"] {{
                background-color: #ffe0b2; /* Light orange */
                border-top-left-radius: 4px; /* Slightly less rounded on the top-left */
                margin-right: 20%; /* Push assistant messages to the left */
                border: 1px solid #ffd54f; /* Subtle border */
            }}

            /* Styling for markdown content within chat messages */
            .chat-message-content p {{
                margin-bottom: 0;
                line-height: 1.6;
                color: #333; /* Darker text for readability */
            }}
            .chat-message-content code {{
                background-color: rgba(0, 0, 0, 0.05);
                border-radius: 4px;
                padding: 2px 4px;
                font-family: monospace;
            }}
            .chat-message-content pre code {{
                display: block;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 8px;
                overflow-x: auto;
            }}

            /* Sidebar styling */
            .st-sidebar {{
                background-color: #e9ecef; /* Slightly darker sidebar background */
                padding-top: 1.5rem;
                padding-left: 1rem;
                padding-right: 1rem;
                border-right: 1px solid #dee2e6; /* Subtle border for separation */
                box-shadow: 2px 0 5px rgba(0,0,0,0.05); /* Subtle shadow */
            }}
            .st-sidebar .st-emotion-cache-h5rpjc {{ /* Target specific div for padding within sidebar */
                padding-top: 1rem;
            }}
            .st-sidebar .st-header {{
                color: #343a40; /* Darker header text */
                margin-bottom: 1rem;
            }}

            /* Main title styling */
            h1 {{
                color: #212529; /* Darker title text */
                text-align: center;
                margin-bottom: 2rem;
                font-size: 2.5rem;
                font-weight: 700;
            }}

            /* File uploader button styling */
            .stFileUploader > div > button {{
                background-color: var(--primary-color);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 1rem;
                cursor: pointer;
                transition: background-color 0.2s ease;
            }}
            .stFileUploader > div > button:hover {{
                background-color: #007bff; /* Slightly darker primary on hover */
            }}

            /* Slider track coloring */
            .stSlider > div > div > div {{ /* Target the slider track */
                background-color: var(--primary-color);
            }}
            /* Slider thumb coloring */
            .stSlider .st-emotion-cache-1f8r0ss {{ /* Specific class for the thumb */
                background-color: var(--primary-color);
                border: 2px solid var(--primary-color);
            }}

            /* Info, success, warning boxes */
            .stAlert {{
                border-radius: 8px;
            }}

            /* Button styling */
            .stButton > button {{
                border-radius: 8px;
                border: 1px solid var(--primary-color);
                color: var(--primary-color);
                background-color: white;
                transition: all 0.2s ease;
                min-height: 2.5rem; /* Ensure consistent button height */
            }}
            .stButton > button:hover {{
                background-color: var(--primary-color);
                color: white;
            }}
            /* Specific styling for the Clear Chat button */
            #root .stButton button[key="clear_chat_button"] {{
                background-color: #f44336; /* Red for clear */
                color: white;
                border: none;
                margin-top: 1rem; /* Space from chat input */
                width: 100%; /* Make it full width below chat input */
            }}
            #root .stButton button[key="clear_chat_button"]:hover {{
                background-color: #d32f2f; /* Darker red on hover */
            }}

        </style>
    """, unsafe_allow_html=True)

    sidebar()  # Render the sidebar UI
    st.title("💬 LLaMate — Offline Chat with LLaMA")  # Display the main title

    # Display an initial welcome message if the chat history is empty
    if not st.session_state.history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Hello! I'm LLaMate, your private offline AI assistant. How can I help you today?")
            st.markdown(
                "You can upload a **TXT/PDF document** for summarization or discussion, or an **image** for visual context, from the sidebar. Or just start typing below!")
            st.markdown(
                "Please note: AI response speed depends on your computer's hardware and the selected AI model (e.g., LLaMA-3 8B, 70B).")  # Added note on speed
            # The specific Ollama warning from here was removed as requested.
            # The warning in chat_ui() will still appear and disable input if Ollama isn't running.

    chat_ui()  # Render the main chat interface

if __name__ == "__main__":
    main()
