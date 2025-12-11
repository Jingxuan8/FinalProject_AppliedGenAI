import io
import os
import re
import tempfile
import whisper
import torch
import numpy as np
import streamlit as st
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from audiorecorder import audiorecorder

from agent_graph.graph import create_graph

# ============================================================ #

st.set_page_config(page_title="Agentic Voice-to-Voice AI Assistant for Product Discovery",
                   page_icon="https://img.icons8.com/ios_filled/1200/ai-chatting.jpg")


@st.cache_resource
def get_openai_client() -> OpenAI:
    return OpenAI()


@st.cache_resource
def get_rag_graph():
    return create_graph()

### not needed anymore 
#@st.cache_resource
#def load_tts_models():
#    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
#    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
#
#    # Example speaker embedding from a public dataset, can change if desired, make sure size is (1,512)
#    speaker_embedding = torch.load("data/speaker_embedding_512_new.pt")
#
#    return processor, model, vocoder, speaker_embedding


# below use whisper model (local downloaded) as transcriber
# can changed to openai transcribing
# or can fetch whisper model through api

# Cache the Whisper model so it loads only once
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


def run_rag_pipeline(user_query: str) -> str:
    # get LangGraph
    graph = get_rag_graph()

    final_state = graph.invoke({"user_query": user_query})


    ## find final_state structure here ###############
    print(final_state)


    answer = final_state.get("final_answer") or ""
    return answer


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
            speed=1.5,  # (optionally speed up)
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

    #if st.button("Transcribe & Ask RAG Agent"): # change to no button?

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
            answer = run_rag_pipeline(user_text) # currently not working, debug, use replacment below to test out rest
            #answer = '''In e-commerce, customers often ask for product recommendations by speaking naturally (e.g., “I need an eco-friendly stainless-steel cleaner under $15”). Traditional chatbots struggle to interpret intent, search private catalogs, check live availability, and answer clearly; especially hands-free.
            #This project delivers a voice-to-voice, multi-agent assistant that understands spoken requests, plans a solution path, retrieves grounded evidence from a private catalog (Amazon Product Dataset 2020), optionally compares with live web results via MCP tools, and replies via TTS with citations and basic safety checks. Orchestration must be implemented with LangGraph.'''
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

    st.subheader("Answer in text:")
    st.write(answer)

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


    # Comparison Table #
    st.markdown("## Product Comparison Table")

    # Example placeholder data (replace with real RAG outputs)
    product_name_1 = "Product A"
    product_price_1 = "$29.99"
    product_desc_1 = "A high-quality item with excellent durability."

    product_name_2 = "Product B"
    product_price_2 = "$19.99"
    product_desc_2 = "Affordable choice with solid performance."

    product_name_3 = "Product C"
    product_price_3 = "$39.99"
    product_desc_3 = "Premium build with advanced features."

    # Build table
    comparison_data = {
        "Name": [product_name_1, product_name_2, product_name_3],
        "Price": [product_price_1, product_price_2, product_price_3],
        "Description": [product_desc_1, product_desc_2, product_desc_3]
    }
    st.table(comparison_data)


    # Reference Links #
    st.markdown("### 🔗 Reference Links")
    st.markdown(
        """
        - https://example.com/productA  
        - https://example.com/productB  
        - https://example.com/productC  
        """
    )


    st.markdown("# Logs")

    # Placeholder logs — replace with actual log
    log_messages = [
        "User asked: 'find me a budget laptop'",
        "Agent retrieved 12 candidate products",
        "Filtered to 3 best matches",
        "Generated final response"
    ]

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
            {'<br>'.join(log_messages)}
        </div>
        """,
        unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
