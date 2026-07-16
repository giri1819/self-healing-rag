from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json

load_dotenv()

critic_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def critic(state: dict):

    answer = state["answer"]
    chunks = "\n\n".join(state["retrieved_chunks"])

    system_prompt = """
You are an expert AI fact-checker.

Your job is to verify whether an answer is supported by the retrieved context.

Rules:

1. Use ONLY the retrieved context.

2. The answer MAY combine information from multiple retrieved chunks.

3. Do NOT require the answer to copy the document word-for-word.

4. If the answer logically combines facts from multiple chunks,
consider it supported.

5. Mark a statement as unsupported ONLY if it introduces:
- facts not present in the context
- invented numbers
- invented rankings
- invented scores
- unsupported recommendations
- unsupported assumptions

Return ONLY valid JSON.

Format:

{
    "verdict":"GROUNDED" | "PARTIALLY_GROUNDED" | "NOT_GROUNDED",
    "supported":[
        "...",
        "..."
    ],
    "unsupported":[
        "...",
        "..."
    ]
}
"""

    human_prompt = f"""
CONTEXT:

{chunks}

ANSWER:

{answer}
"""

    response = critic_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
    )

    content = response.content.strip()

    try:
        report = json.loads(content)

    except Exception:

        report = {
            "verdict": "NOT_GROUNDED",
            "supported": [],
            "unsupported": [
                "Critic could not parse the response."
            ]
        }

    verdict = report["verdict"]

    print("\n========== CRITIC REPORT ==========")
    print(json.dumps(report, indent=4))
    print("===================================\n")

    return {
        **state,
        "critic_report": report,
        "is_grounded": verdict == "GROUNDED",
        "final_answer":
            answer if verdict in
            ["GROUNDED", "PARTIALLY_GROUNDED"]
            else None
    }