import base64
import html
from pathlib import Path

import streamlit as st

from services.modal_api import chat_with_bot, transcribe_audio
from services.tts import synthesize_edge_tts


st.set_page_config(
    page_title="SunaSunau",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: "Space Grotesk", system-ui, -apple-system, sans-serif;
        letter-spacing: 0.1px;
    }

    .stApp {
        background: radial-gradient(1200px 600px at 20% -10%, #1b242f 0%, #0b0f14 60%);
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(217, 119, 62, 0.4);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }

    .demo-link-wrap {
        display: flex;
        justify-content: center;
        margin: 2px 0 12px 0;
    }

    .demo-link {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.72rem;
        line-height: 1.2;
        color: rgba(200, 194, 184, 0.7) !important;
        background: transparent;
        border: none;
        box-shadow: none;
        text-decoration: underline;
        text-underline-offset: 2px;
    }

    .demo-link:hover {
        color: rgba(200, 194, 184, 0.95) !important;
        text-decoration: underline;
    }

    div[data-testid="stMetric"] {
        background: #131a22;
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 12px;
        padding: 10px 12px;
    }

    .stCaption {
        color: #c8c2b8;
    }

    .reply-card {
        background: linear-gradient(135deg, rgba(19, 26, 34, 0.95), rgba(27, 36, 47, 0.95));
        border: 1px solid rgba(217, 119, 62, 0.2);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
    }

    .reply-label {
        color: #d9773e;
        font-size: 0.85rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .reply-text {
        color: #e8e3da;
        font-size: 1.05rem;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    .title-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
    }

    .title-row h1 {
        margin: 0;
        line-height: 1.1;
    }

    .info-tooltip {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 19px;
        height: 19px;
        margin-top: 8px;
        border-radius: 999px;
        border: 1px solid rgba(217, 119, 62, 0.5);
        color: #d9773e;
        font-size: 12px;
        font-weight: 600;
        line-height: 1;
        cursor: help;
    }

    .info-tooltip:hover .tooltip-text {
        opacity: 1;
        transform: translate(-50%, 0);
        pointer-events: auto;
    }

    .tooltip-text {
        position: absolute;
        left: 50%;
        top: 28px;
        transform: translate(-50%, -6px);
        width: min(340px, 82vw);
        background: rgba(19, 26, 34, 0.98);
        border: 1px solid rgba(217, 119, 62, 0.25);
        color: #e8e3da;
        font-size: 0.85rem;
        font-weight: 400;
        line-height: 1.5;
        padding: 10px 12px;
        border-radius: 10px;
        box-shadow: 0 16px 30px rgba(0, 0, 0, 0.35);
        opacity: 0;
        transition: opacity 0.2s ease, transform 0.2s ease;
        pointer-events: none;
        z-index: 999;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="title-row">
        <h1>SunaSunau</h1>
        <div class="info-tooltip" aria-label="Usage tips">
            i
            <div class="tooltip-text">
                Speak clearly and slowly in Nepali for the best results.
                You can ask general knowledge questions about topics like history,
                the constitution, and current affairs, or have an emotional conversation.
                Replies may take around 10 seconds to 1 minute depending on the request.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "AI-powered Nepali voice assistant for speech recognition, punctuation restoration, and natural chatbot conversations."
)


DEMO_AUDIO_PATH = Path(__file__).with_name("demo.wav")


if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

if "active_audio_source" not in st.session_state:
    st.session_state.active_audio_source = None
    # values: None, "demo", "recorded"

if "audio_recorder_key" not in st.session_state:
    st.session_state.audio_recorder_key = 0

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "reply" not in st.session_state:
    st.session_state.reply = ""

if "scroll_to_chat" not in st.session_state:
    st.session_state.scroll_to_chat = False


# Handle demo link click.
if st.query_params.get("demo") == "1":
    if DEMO_AUDIO_PATH.exists():
        st.session_state.audio_bytes = DEMO_AUDIO_PATH.read_bytes()
        st.session_state.active_audio_source = "demo"

        # Reset recorder so old recorded audio does not override demo audio.
        st.session_state.audio_recorder_key += 1

        st.query_params.clear()
        st.rerun()
    else:
        st.query_params.clear()
        st.error("demo.wav not found in the app folder.")


with st.container(border=True):
    st.subheader("Speak")

    st.markdown(
        """
        <div class="demo-link-wrap">
            <a class="demo-link" href="?demo=1" target="_self">▶ Try Demo Speech</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.active_audio_source == "demo" and st.session_state.audio_bytes:
        st.success("Demo audio loaded. You can transcribe now.")
        st.caption("Demo audio")
        st.audio(st.session_state.audio_bytes, format="audio/wav")

    audio_file = st.audio_input(
        "Tap the mic to record",
        key=f"audio_recorder_{st.session_state.audio_recorder_key}",
    )

    if audio_file is not None:
        previous_source = st.session_state.active_audio_source

        st.session_state.audio_bytes = audio_file.read()
        st.session_state.active_audio_source = "recorded"

        st.success("Audio captured. You can transcribe now.")

        # If demo player was visible, rerun once to remove it immediately.
        if previous_source == "demo":
            st.rerun()

    if st.session_state.audio_bytes is None:
        st.info("Waiting for audio. Tap the mic or run the demo.")
    elif st.session_state.active_audio_source == "recorded":
        st.caption("Recorded audio selected.")

    can_transcribe = st.session_state.audio_bytes is not None

    if st.button(
        "Transcribe using SunaSunau",
        use_container_width=True,
        disabled=not can_transcribe,
    ):
        with st.spinner("Transcribing..."):
            try:
                result = transcribe_audio(st.session_state.audio_bytes)
                st.session_state.transcript = result.get("text", "")
                st.session_state.scroll_to_chat = True
            except Exception as exc:
                st.error(f"Transcribe failed: {exc}")


st.markdown('<div id="chat-section"></div>', unsafe_allow_html=True)


with st.container(border=True):
    st.subheader("Chat")

    user_text = st.text_area(
        "Your Message:",
        value=st.session_state.transcript,
        height=120,
        placeholder="Your transcribed text will appear. You may also write on your own.",
    )

    mode = st.radio(
        "Answer style",
        ["factual", "emotional"],
        horizontal=True,
    )

    if st.button(
        "Talk with SunaSunau",
        type="primary",
        use_container_width=True,
    ):
        if not user_text.strip():
            st.error("Please enter a message or transcribe audio before talking with SunaSunau.")
        else:
            with st.spinner("Thinking..."):
                try:
                    reply = chat_with_bot(user_text.strip(), mode)
                    st.session_state.reply = reply.get("text", "")
                except Exception as exc:
                    st.error(f"Chat failed: {exc}")

    if st.session_state.scroll_to_chat:
        st.markdown(
            """
            <script>
            const chatSection = parent.document.querySelector('#chat-section');
            if (chatSection) {
                chatSection.scrollIntoView({ behavior: 'smooth' });
            }
            </script>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.scroll_to_chat = False

    if st.session_state.reply:
        safe_reply = html.escape(st.session_state.reply)

        st.markdown(
            f"""
            <div class="reply-card">
                <div class="reply-label">SunaSunau Reply</div>
                <div class="reply-text">{safe_reply}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Generating speech..."):
            try:
                audio_bytes = synthesize_edge_tts(st.session_state.reply)
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

                st.markdown(
                    f"""
                    <audio autoplay style="display:none;">
                        <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg" />
                    </audio>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"TTS failed: {exc}")