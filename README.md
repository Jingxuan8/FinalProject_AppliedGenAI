# FinalProject_AppliedGenAI

# FinalProject_AppliedGenAI

## 1. Overview

This project implements a **voice-to-voice multi-agent assistant** built with **LangGraph**, **custom MCP tools**, and a **Streamlit UI**.  
The system accepts spoken queries, transcribes them, performs retrieval-augmented reasoning through a multi-agent pipeline, and replies with synthesized speech while displaying structured results.

The assistant supports:
- Price checking  
- Availability checking  
- Product search & recommendations  
- Safety filtering  
- RAG + Web hybrid retrieval  
- Voice input and voice output  

---

## 2. Overall Architecture

The system follows a **three-layer architecture** that cleanly separates interaction, reasoning, and tooling.

At the top layer, the **Streamlit UI** handles microphone input, automatic speech recognition (ASR), transcript display, and text-to-speech (TTS) playback. The recognized `user_text` is passed into the **LangGraph pipeline**, which acts as the core decision engine. Inside this pipeline, the **Router** interprets the query and sets intent, the **Planner** determines which tools to activate, the **Retriever** gathers evidence from both RAG and web sources, and the **Answerer** generates the final responses. All agents operate over a shared JSON state that flows through the graph.

Whenever external information is required, the pipeline issues tool calls to the **MCP Server**, which hosts structured tools such as `web_search` and `rag_search`. The MCP server returns normalized results back into the agent pipeline, completing a closed feedback loop.

This layered design makes the system modular, debuggable, and easy to extend.


---

## 4. Graph Design

The system is structured as a **multi-agent LangGraph pipeline** where each agent operates over a shared JSON state. Each agent reads the state, appends its own outputs, and passes it forward without changing the schema.

The pipeline flow is:
1. **Router** – interprets the user query and sets intent, constraints, and a safety flag.  
2. **Planner** – decides which retrieval modes to activate (RAG, Web, or both).  
3. **Retriever** – executes tool calls and populates the state with RAG results, web results, prices, and availability.  
4. **Answerer** – cleans and ranks results, applies safety overrides if needed, and generates both paper (text) and speech outputs.

This design keeps all agents **stateless**, transparent, and reproducible. The shared JSON state acts as the single source of truth across the entire graph.


---

## 5. MCP Server & Tool Schemas

External capabilities are integrated through a custom **Model Context Protocol (MCP) server**. The server exposes two primary tools with strict JSON schemas:

### web_search
- Wraps live HTTP-based search
- Inputs: `query`, optional `num_results`
- Outputs: `title`, `url`, `snippet`, `price` (nullable), availability (inferred later)

### rag_search
- Queries a local FAISS/Chroma vector store
- Inputs: `query`, optional filters (`category`, `max_price`)
- Outputs: structured product metadata with stable identifiers

Both tools are schema-validated before execution, guaranteeing predictable and safe agent-tool interaction. Tool outputs are normalized so downstream agents can treat RAG and web results uniformly.

All tools are registered in a single MCP `server.py`, ensuring centralized control and extensibility.

---

## 6. Safety Notes

Safety is enforced at the **Router** stage. If a query is classified as unsafe, the router sets `safety_flag = True` in the shared state. The **Answerer** checks this flag before any generation and immediately returns a controlled refusal message, skipping all retrieval and reasoning.

Additional safeguards include:
- No exposure of raw URLs in speech output  
- Removal of sensitive or identifying information  
- Prevention of operational or harmful instructions  

Safety is enforced both at query interpretation and answer generation time.

---

## 7. Setup Instructions

Before running the system, configure environment variables.

1. Copy `.env.example` to `.env` in the project root.
2. Fill in:
   - `OPENAI_API_KEY` — required for LLM reasoning  
   - `SERPER_API_KEY` — required for web search  
   - `MODEL_NAME` (optional) — swap models without code changes  

Install dependencies:

pip install -r requirements.txt


## 8. Run Scripts

The system requires **two concurrent processes**: the MCP tool server and the Streamlit UI. These must be started in separate terminals from the project root directory.

### 8.1 Start MCP Server (Tool Backend)

The MCP server exposes structured tools (such as `web_search` and `rag_search`) to the LangGraph agent pipeline.

```bash
python -m mcp_server.server --transport http --host localhost --port 8765
