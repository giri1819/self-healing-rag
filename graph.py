from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict

from nodes.retriever import retrieve
from nodes.generator import generate
from nodes.critic import critic
from nodes.reformulator import reformulate


class RAGState(TypedDict):

    question: str

    retrieved_chunks: List[str]

    answer: str

    is_grounded: bool

    retry_count: int

    final_answer: Optional[str]

    critic_report: Dict


def should_retry(state: RAGState):

    verdict = state["critic_report"]["verdict"]

    if verdict == "GROUNDED":

        return "end"

    elif verdict == "PARTIALLY_GROUNDED":

        return "partial"

    elif state["retry_count"] >= 2:

        return "give_up"

    else:

        return "reformulate"


def build_graph(retriever):

    graph = StateGraph(RAGState)

    # ---------------- Nodes ----------------

    graph.add_node(
        "retrieve",
        lambda s: retrieve(s, retriever)
    )

    graph.add_node(
        "generate",
        generate
    )

    graph.add_node(
        "critic",
        critic
    )

    graph.add_node(
        "reformulate",
        reformulate
    )

    graph.add_node(
        "partial",
        lambda s: {
            **s,
            "final_answer": s["answer"]
        }
    )

    graph.add_node(
        "give_up",
        lambda s: {
            **s,
            "final_answer":
            "The uploaded document does not contain enough information to answer this question accurately."
        }
    )

    # ---------------- Flow ----------------

    graph.set_entry_point("retrieve")

    graph.add_edge(
        "retrieve",
        "generate"
    )

    graph.add_edge(
        "generate",
        "critic"
    )

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "end": END,
            "partial": "partial",
            "reformulate": "reformulate",
            "give_up": "give_up"
        }
    )

    graph.add_edge(
        "reformulate",
        "retrieve"
    )

    graph.add_edge(
        "partial",
        END
    )

    graph.add_edge(
        "give_up",
        END
    )

    return graph.compile()