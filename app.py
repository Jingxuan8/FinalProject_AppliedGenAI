import io
import os
import re
import tempfile
import whisper
import torch
import numpy as np
import pandas as pd
import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from audiorecorder import audiorecorder

from agent_graph.graph import create_graph


from mcp_server.utils.logger import LOG_FILE
if "log_start_pos" not in st.session_state:
    try:
        st.session_state["log_start_pos"] = os.path.getsize(LOG_FILE)
    except:
        st.session_state["log_start_pos"] = 0

def read_new_logs():
    log_path = LOG_FILE

    if not os.path.exists(log_path):
        return ["Log file does not exist."]

    logs = []
    start = st.session_state.get("log_start_pos", 0)

    with open(log_path, "r", encoding="utf-8") as f:
        f.seek(start)  # jump to where this session started
        for line in f:
            logs.append(line.rstrip())

    return logs


# ============================================================ #

st.set_page_config(page_title="Agentic Voice-to-Voice AI Assistant for Product Discovery",
                   page_icon="https://img.icons8.com/ios_filled/1200/ai-chatting.jpg")
st.markdown("""
    <style>
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 80% !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI()


@st.cache_resource
def get_rag_graph():
    return create_graph()


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("data/whisper/small.pt")


def transcribe_audio_to_text(file_bytes: bytes, filename: str) -> str:
    # Extract end if file extension (file type)
    if "." in filename:
        ext = filename.split(".")[-1]
        suffix = f".{ext}"
    else:
        suffix = ".wav"

    # Save uploaded bytes to a temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        audio_path = tmp.name

    # Load model
    model = load_whisper_model()

    # Run Whisper transcription
    result = model.transcribe(audio_path)
    return result.get("text", "")

def safe_price(value):
    try:
        return f"{float(value):.2f}"
    except:
        return "N/A"

def run_rag_pipeline(user_query: str):
    # get LangGraph
    graph = get_rag_graph()

    final_state = graph.invoke({"user_query": user_query})
    if final_state.get("safety_flag", False) is True:
        answer = final_state.get("speech_answer") or ""
        paper_answer = final_state.get("paper_answer") or ""
        comparison_data = {
        "Name": ["","",""],
        "Price ($)": ["","",""],
        "Reference Link": ["","",""],
        "Doc ID": ["","",""],
        }
        df = pd.DataFrame(comparison_data)
        debug_log = final_state.get("debug_log", [])
        return answer, df, paper_answer, debug_log

    # pick top 3 and extract
    max_items = max(3, len(final_state.get("selected_items")))

    product_names = []
    product_prices = []
    product_urls = []
    product_docs = []

    for i in range(max_items):
        product_names.append(final_state.get("selected_items")[i]["title"])
        product_prices.append(safe_price(final_state.get('selected_items')[i]['price']))
        product_urls.append(final_state.get("selected_items")[i]["url"])
        product_docs.append(final_state.get("selected_items")[i]["doc_id"])

    comparison_data = {
        "Name": product_names,
        "Price ($)": product_prices,
        "Reference Link": product_urls,
        "Doc ID": product_docs,
    }

    df = pd.DataFrame(comparison_data)
    df.index = df.index + 1

    answer = final_state.get("speech_answer") or ""
    paper_answer = final_state.get("paper_answer") or ""
    debug_log = final_state.get("debug_log", [])
    return answer, df, paper_answer, debug_log


# fragmented tts
def split_into_chunks(text: str, max_chars: int = 180):
    parts = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    curr = ""

    for p in parts:
        if not p.strip():
            continue
        candidate = curr + p
        if len(candidate) <= max_chars:
            curr = candidate
        else:
            if curr:
                chunks.append(curr.strip())
            curr = p

    if curr:
        chunks.append(curr.strip())

    return chunks


# use gpt-4o-mini-tts, can switch to other model (remember put openai api key)
def synthesize_answer_to_wav(answer_text: str) -> bytes:
    client = get_openai_client()

    # Split into fragments for more stable prosody
    chunks = split_into_chunks(answer_text)  
    pcm_segments = []

    for chunk in chunks:
        # Request a WAV buffer from OpenAI TTS per chunk
        res = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="nova", # avalible: 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'coral', 'verse', 'ballad', 'ash', 'sage', 'marin', and 'cedar'
            input=chunk,
            speed=1.3,  # (optionally speed up)
            response_format="wav"
        )

        wav_bytes = res.read()

        # Decode WAV → PCM
        audio, sr = sf.read(io.BytesIO(wav_bytes))

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:  
            audio = np.mean(audio, axis=1)

        pcm_segments.append(audio)

    # Concatenate all fragments
    if len(pcm_segments) == 0:
        return b""

    full_pcm = np.concatenate(pcm_segments, axis=0)

    # Re-encode PCM → WAV
    out_buf = io.BytesIO()
    sf.write(out_buf, full_pcm, samplerate=16000, format="WAV")
    out_buf.seek(0)

    return out_buf.getvalue()


def main():
    # style
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(120deg, #f5f7fa 0%, #c3cfe2 100%);
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Agentic Voice-to-Voice AI Assistant for Product Discovery")
    st.markdown("### 🎙️ Ask your question")

    # Fragment based recording by audiorecorder, better quality
    audio_segment = audiorecorder("Start recording", "Stop recording")

    recorded_audio_bytes = None
    if len(audio_segment) > 0:
        # Preview recording
        st.audio(audio_segment.export().read(), format="audio/wav")

        # Export to WAV bytes for Whisper
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        recorded_audio_bytes = wav_io.getvalue()


    if recorded_audio_bytes is None:
        st.info("Go ahead and record your question!")
        return

    raw_bytes = recorded_audio_bytes
    filename = "recording.wav"

    with st.spinner("Transcribing audio..."):
        try:
            user_text = transcribe_audio_to_text(raw_bytes, filename)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return

    st.subheader("Recognized question:")
    st.write(user_text)

    with st.spinner("Querying AI assistant..."):
        try:
            answer, df, paper_answer, agent_log = run_rag_pipeline(user_text)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

    st.subheader("Answer in text:")
    st.write(paper_answer) # can change to answer if you want just voice transcription

    with st.spinner("Synthesizing spoken answer into voice response..."):
        try:
            wav_bytes = synthesize_answer_to_wav(answer)
        except Exception as e:
            st.error(f"TTS synthesis failed: {e}")
            return

    if wav_bytes:
        st.subheader("Audio summary:")
        st.audio(wav_bytes, format="audio/wav")

        st.download_button(
            label="Download answer_output.wav",
            data=wav_bytes,
            file_name="answer_output.wav",
            mime="audio/wav",
        )
    else:
        st.warning("No audio was generated from the answer text.")

    #log
    new_logs = read_new_logs()
    try:
        st.session_state["log_start_pos"] = os.path.getsize(LOG_FILE)
    except:
        st.session_state["log_start_pos"] = 0


    # Comparison Table #
    st.markdown("## Product Comparison Table")

    # Build table
    st.markdown("""
    <style>
    /* Make table full width */
    table {
        width: 100% !important;
    }

    /* Header styling */
    thead th {
        background-color: #f0f0f0 !important;
        color: black !important;
        font-weight: 800 !important;   /* header bold */
        font-size: 16px !important;
        border: 2px solid black !important;
        padding: 10px !important;
        white-space: nowrap !important;
    }

    /* Rows */
    tbody td {
        border: 2px solid black !important;
        padding: 12px !important;
        font-size: 15px !important;
    }

    /* Index column (left-most) */
    tbody th {
        font-weight: 800 !important;   /* bold index */
        color: black !important;        /* black text */
        border: 2px solid black !important;
        padding: 10px !important;
        background-color: #fafafa !important;  /* subtle background */
        font-size: 15px !important;
    }
                    
    /* Alternating row colors */
    tbody tr:nth-child(odd) {
        background-color: #fafafa !important;
    }

    tbody tr:nth-child(even) {
        background-color: #f2f2f2 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display table
    st.table(df)


    st.markdown("### Logs")
    agent_log = agent_log or []
    new_logs = new_logs or []
    combined_logs = []
    for entry in agent_log:
        combined_logs.append(str(entry))
    for entry in new_logs:
        combined_logs.append(str(entry))

    # scrollable log box
    st.markdown(
        f"""
        <div style="
            background-color:#f5f5f5;
            padding:12px;
            border-radius:8px;
            height:200px;
            overflow-y:auto;
            border:1px solid #ccc;
            font-size:14px;
            line-height:1.4;
        ">
            {'<br>'.join(combined_logs)}
        </div>
        """,
        unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
