"""A compact, inspectable Knowledge-Graph RAG implementation for beginners."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import networkx as nx
from openai import OpenAI

STOP_WORDS = {
    "about", "after", "and", "are", "for", "from", "has", "have", "into", "its",
    "more", "not", "the", "this", "that", "their", "with", "what", "which", "will",
    "scheme", "schemes", "tamil", "nadu", "farmer", "farmers",
}


def tokens(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z]{4,}", text) if word.lower() not in STOP_WORDS}


def load_schemes(path: str = "data/schemes.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_graph(schemes: list[dict]) -> nx.Graph:
    """Nodes are schemes, department and meaningful shared topic words."""
    graph = nx.Graph()
    for scheme in schemes:
        sid = scheme["id"]
        graph.add_node(sid, kind="scheme", label=scheme["name"], record=scheme)
        department = scheme["department"]
        graph.add_node(department, kind="department", label=department)
        graph.add_edge(sid, department, relation="OPERATED_BY")
        # Limit topic nodes so generic words do not drown out useful links.
        for topic, count in Counter(tokens(scheme["name"] + " " + scheme["text"])).most_common(20):
            if count < 2 and topic not in tokens(scheme["name"]):
                continue
            topic_id = f"topic:{topic}"
            graph.add_node(topic_id, kind="topic", label=topic)
            graph.add_edge(sid, topic_id, relation="MENTIONS")
    return graph


def retrieve(question: str, graph: nx.Graph, limit: int = 4) -> list[dict]:
    """Graph-aware retrieval: match topics, then return neighbouring scheme nodes."""
    query = tokens(question)
    scores: Counter[str] = Counter()
    for node, attributes in graph.nodes(data=True):
        if attributes.get("kind") != "scheme":
            continue
        record = attributes["record"]
        scheme_words = tokens(record["name"] + " " + record["text"])
        scores[node] += 3 * len(query & tokens(record["name"])) + len(query & scheme_words)

    # A mentioned topic points to all related schemes in the knowledge graph.
    for word in query:
        topic_id = f"topic:{word}"
        if graph.has_node(topic_id):
            for neighbour in graph.neighbors(topic_id):
                if graph.nodes[neighbour].get("kind") == "scheme":
                    scores[neighbour] += 2

    ranked = [node for node, score in scores.most_common(limit) if score > 0]
    return [graph.nodes[node]["record"] for node in ranked]


def context_for(records: list[dict]) -> str:
    blocks = []
    for record in records:
        blocks.append(
            f"SCHEME: {record['name']}\nDEPARTMENT: {record['department']}\n"
            f"OFFICIAL SOURCE: {record['source_url']}\nDETAILS: {record['text'][:3500]}"
        )
    return "\n\n---\n\n".join(blocks)


def answer(question: str, records: list[dict]) -> str:
    """Generate only from retrieved official text; gracefully work without an API key."""
    if not records:
        return "I could not find a matching scheme in the downloaded official pages. Try different keywords or refresh the data."
    if not os.getenv("OPENAI_API_KEY"):
        names = "\n".join(f"- {item['name']}" for item in records)
        return "Add OPENAI_API_KEY to .env for a natural-language answer. The graph retrieved:\n" + names

    prompt = f"""You are a careful assistant for Tamil Nadu farmer-welfare schemes.
Answer the user's question using ONLY the retrieved official text below. If it lacks an answer, say so plainly.
Do not invent eligibility, subsidy amounts, application dates, or contacts. Be concise and end with the official source URL(s).

QUESTION: {question}

RETRIEVED OFFICIAL TEXT:
{context_for(records)}"""
    client = OpenAI()
    response = client.responses.create(
        model=os.getenv("CHAT_MODEL", "gpt-5-mini"),
        input=prompt,
        store=False,
    )
    return response.output_text
