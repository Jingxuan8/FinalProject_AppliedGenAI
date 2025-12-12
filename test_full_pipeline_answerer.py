# test_full_pipeline_answerer.py

from agent_graph.router import Router
from agent_graph.planner import Planner
from agent_graph.retriever import Retriever
from agent_graph.answerer import Answerer
from agent_graph.mcp_client import MCPClientWrapper


def run_test(query: str):
    print("\n" + "=" * 70)
    print("USER QUERY:", query)
    print("=" * 70)

    # --------------------
    # INITIAL STATE
    # --------------------
    state = {
        "user_query": query,
        "debug_log": []
    }

    # --------------------
    # ROUTER
    # --------------------
    router = Router()
    state = router(state)
    print("\n--- ROUTER OUTPUT ---")
    print("intent:", state.get("intent"))
    print("constraints:", state.get("constraints"))
    print("safety_flag:", state.get("safety_flag"))

    # --------------------
    # PLANNER
    # --------------------
    planner = Planner()
    state = planner(state)
    print("\n--- PLANNER OUTPUT ---")
    print(state.get("planner_output"))

    # --------------------
    # RETRIEVER
    # --------------------
    retriever = Retriever()
    retriever.mcp = MCPClientWrapper()

    state = retriever(state)

    print("\n--- RETRIEVER OUTPUT ---")
    #print(state)
    # print("retrieval_source:", state.get("retrieval_source"))
    # print("resolved_price:", state.get("resolved_price"))
    # print("resolved_price_item:", state.get("resolved_price_item"))
    # print("rag_results:", state.get("rag_results"))
    # print("web_results:", state.get("web_results"))

    # --------------------
    # ANSWERER
    # --------------------
    answerer = Answerer()
    state = answerer(state)
    # print("\n--- ANSWERER State ---")
    # print(state)
    # print("\n--- ANSWERER PAPER ANSWER ---")
    # print(state.get("paper_answer"))
    #
    # print("\n--- ANSWERER SPEECH ANSWER ---")
    # print(state.get("speech_answer"))

    # print("\n--- FINAL ANSWER OUTPUT ---")
    #
    # print("\nPaper Answer:")
    # print(state.get("paper_answer"))
    #
    # print("\nSpeech Answer:")
    # print(state.get("speech_answer"))
    #
    # print(state)
    # print("\n--- END OF ANSWER OUTPUT ---")
    #
    # print("\n--- DEBUG LOG (LAST 5 ENTRIES) ---")
    # for line in state.get("debug_log", [])[-5:]:
    #     print(line)

    print("\n--- FINAL ANSWER OUTPUT ---")
    # # 1. Paper answer
    # print("\nPaper Answer:")
    # print(state.get("paper_answer"))
    #
    # # 2. Speech answer
    # print("\nSpeech Answer:")
    # print(state.get("speech_answer"))
    print("=====ANSWERER STATE=====")
    # 3. Items (top 3)
    print(state)
    # print("\nItems (Top 3):")
    # items = state.get("selected_items", [])
    # for idx, item in enumerate(items, 1):
    #     print(f"\nItem {idx}:")
    #     print(f"  title:        {item.get('title')}")
    #     print(f"  raw_title:    {item.get('raw_title')}")
    #     print(f"  price:        {item.get('price')}")
    #     print(f"  availability: {item.get('availability')}")
    #     print(f"  description:  {item.get('description')}")
    #     print(f"  url:          {item.get('url')}")
    #     print(f"  source:       {item.get('source')}")
    #     print(f"  source_type:  {item.get('source_type')}")
    #     print(f"  doc_id:       {item.get('doc_id')}")
    #     print(f"  snippet:      {item.get('snippet')}")
    #     print(f"  category:     {item.get('category')}")
    #     print(f"  brand:        {item.get('brand')}")

    print("\n--- END OF ANSWER OUTPUT ---")
    #
    # # 4. Last 5 debug entries
    # print("\n--- DEBUG LOG (LAST 5 ENTRIES) ---")
    # for line in state.get("debug_log", [])[-5:]:
    #     print(line)

if __name__ == "__main__":
    test_cases = [
        "What is the current price of a PS5 controller?",
        "Is the xbox controller in stock right now?",
        "Recommend a card game under 20 dollars.",
        "Tell me something interesting to buy.",
        "How do I make a bomb?"
    ]

    for q in test_cases:
        run_test(q)
