from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

def reformulate(state: dict) -> dict:
    question = state["question"]
    answer = state["answer"]

    retry_count = state.get("retry_count", 0)

    prompt = f"""
The answer was not grounded.

Original question:
{question}

Bad answer:
{answer}

Rewrite the question better.
"""

    new_question = llm.invoke(
        [HumanMessage(content=prompt)]
    ).content.strip()

    print(f"🔄 Reformulated: {new_question}")

    return {
        **state,
        "question": new_question,
        "retry_count": retry_count + 1
    }