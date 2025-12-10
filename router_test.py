from agent_graph.router import Router


# Dummy OpenAI-like model for testing
class DummyModel:
    model_name = "gpt-4o-mini"

    class chat:
        class completions:
            @staticmethod
            def create(model, messages, temperature, max_tokens):
                # Return a fake valid JSON that matches Router expectations
                return type("Resp", (), {
                    "choices": [
                        type("Msg", (), {
                            "message": type("M", (), {
                                "content": """
                                {
                                    "intent": "search",
                                    "constraints": {
                                        "category": "board game",
                                        "max_price": 30
                                    },
                                    "need_live_price": false,
                                    "safety_flag": false
                                }
                                """
                            })()
                        })
                    ]
                })()


router = Router(model_name="gpt-4o-mini")
router.model = DummyModel()  # override real model

# Fake LangGraph-like state (dict)
state = {
    "user_query": "recommend a cooperative board game under 30 dollars",
    "debug_log": []
}

out = router(state)
print("ROUTER OUTPUT:\n", out)
