import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


import streamlit as st


load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def generate(state: dict) -> dict:

    question = state["question"]
    chunks = "\n\n".join(state["retrieved_chunks"])

    system_prompt = """
You are an expert document analysis assistant.

STRICT RULES:

1. Answer ONLY using the retrieved context.

2. NEVER invent facts.

3. NEVER invent rankings.

4. NEVER invent scores like 8/10.

5. NEVER assume information.

6. If the document does not contain enough information,
say exactly:

'The document does not contain enough information to answer this question.'

7. If the question asks for comparison:

- Compare every project found in the retrieved context.
- Use only evidence from the document.
- If one project is missing from the retrieved context,
mention that instead of guessing.

8. If the user asks for recommendation:

Recommend ONLY if the document contains enough evidence.

Otherwise state that more information is required.

9. Never use outside knowledge.
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

    print(f"💬 Generated answer: {response.content[:100]}...")

    return {
        **state,
        "answer": response.content
    }