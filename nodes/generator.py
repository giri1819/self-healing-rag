import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

print(os.getenv("GROQ_API_KEY"))
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=api_key
    )


def generate(state: dict) -> dict:

    llm = get_llm()

    question = state["question"]
    chunks = "\n\n".join(state["retrieved_chunks"])

    system_prompt = """
    ...
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"""
Retrieved Context:

{chunks}

Question:

{question}

Answer:
"""
        )
    ]

    response = llm.invoke(messages)

    return {
        **state,
        "answer": response.content
    }
