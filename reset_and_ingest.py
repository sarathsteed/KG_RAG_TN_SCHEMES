import os
os.environ["XXHASH_FORCE_PURE_PYTHON"] = "1"

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

load_dotenv()

# 1. Connect to Neo4j and CLEAR EVERYTHING
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

print("Clearing old data from Neo4j database...")
graph.query("MATCH (n) DETACH DELETE n")
print("Database cleared!")

# 2. Scrape TN Government Scheme List
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Fetching latest web data...")
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

scheme_docs = []

# Exclude non-scheme navigation elements
ignore_terms = [
    "screen reader", "தமிழ்", "home", "contact", "department", 
    "back", "next", "search", "skip to", "sitemap", "feedback"
]

for link in soup.find_all("a"):
    title = link.text.strip()
    href = link.get("href", "")
    
    if title and len(title) > 10:
        if not any(ignore in title.lower() for ignore in ignore_terms):
            doc_content = f"Scheme Name: {title}. Managed by Agriculture Department, Tamil Nadu Government."
            scheme_docs.append(Document(page_content=doc_content))

# Remove duplicate entries
unique_docs = {doc.page_content: doc for doc in scheme_docs}.values()
scheme_docs = list(unique_docs)

print(f"Scraped {len(scheme_docs)} clean scheme titles.")

# 3. Transform to Graph Nodes
print("Extracting Graph Nodes using LLM...")
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Scheme", "Department"],
    allowed_relationships=["MANAGED_BY"]
)

graph_documents = transformer.convert_to_graph_documents(scheme_docs)

# 4. Save to Neo4j and map 'id' property to 'name' for Cypher compatibility
print("Saving nodes to Neo4j...")
graph.add_graph_documents(graph_documents)

# Map node properties so 'name' is populated
graph.query("MATCH (s:Scheme) SET s.name = s.id")

node_count = graph.query("MATCH (s:Scheme) RETURN count(s) AS count")[0]["count"]
print(f"\nSUCCESS! Total clean Scheme nodes currently in Neo4j: {node_count}")