import os
os.environ["XXHASH_FORCE_PURE_PYTHON"] = "1"

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

load_dotenv()

# 1. Connect to Neo4j and Clear Old Data
print("Connecting to Neo4j...")
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

print("Clearing old graph nodes...")
graph.query("MATCH (n) DETACH DELETE n")

# 2. Scrape Scheme List + Deep Link Parsing
base_url = "https://www.tn.gov.in"
list_url = f"{base_url}/scheme_list.php?dep_id=Mg=="
headers = {"User-Agent": "Mozilla/5.0"}

print("Scraping main scheme list page...")
response = requests.get(list_url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

scheme_documents = []
ignore_terms = ["screen reader", "தமிழ்", "home", "contact", "department", "back", "next", "sitemap"]

# Find scheme table links
for link in soup.find_all("a"):
    title = link.text.strip()
    href = link.get("href", "")
    
    if title and len(title) > 10 and not any(term in title.lower() for term in ignore_terms):
        detailed_text = f"Scheme Name: {title}. Managed by Agriculture Department, Tamil Nadu."
        
        # Follow deep link if available to collect page body
        if href and "scheme_view.php" in href or "dept_page" in href:
            full_link = href if href.startswith("http") else f"{base_url}/{href}"
            try:
                detail_resp = requests.get(full_link, headers=headers, timeout=5)
                detail_soup = BeautifulSoup(detail_resp.content, "html.parser")
                # Collect text from target content container
                page_text = " ".join([p.text.strip() for p in detail_soup.find_all(["p", "td", "li"]) if len(p.text.strip()) > 20])
                if page_text:
                    detailed_text += f" Details and Eligibility: {page_text[:1200]}"
            except Exception:
                pass
                
        scheme_documents.append(Document(page_content=detailed_text))

# Remove duplicates
unique_docs = {doc.page_content: doc for doc in scheme_documents}.values()
scheme_documents = list(unique_docs)
print(f"Collected {len(scheme_documents)} scheme pages with deep content.")

# 3. Extract Knowledge Graph Entities
print("Extracting Graph Nodes & Relationships using LLM...")
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Scheme", "Beneficiary", "Benefit", "Requirement", "Department"],
    allowed_relationships=["MANAGED_BY", "PROVIDES", "TARGETS", "REQUIRES"]
)

graph_documents = transformer.convert_to_graph_documents(scheme_documents)

# 4. Save to Neo4j and Normalize Properties
print("Saving detailed nodes to Neo4j...")
graph.add_graph_documents(graph_documents)

# Ensure 'name' property is populated for Cypher compatibility
graph.query("MATCH (s:Scheme) SET s.name = s.id")

node_count = graph.query("MATCH (s:Scheme) RETURN count(s) AS count")[0]["count"]
print(f"\nSUCCESS! Rebuilt Knowledge Graph with {node_count} detailed scheme nodes.")