import os
# Prevent xxhash DLL security block on Windows
os.environ["XXHASH_FORCE_PURE_PYTHON"] = "1"

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

load_dotenv()

# 1. Scrape Tamil Nadu Schemes Page
url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Fetching webpage...")
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

scheme_docs = []

# Broader link & table parsing to ensure scheme names are captured
for link in soup.find_all("a"):
    title = link.text.strip()
    href = link.get("href", "")
    
    # Filter for scheme-related links and text length
    if title and "scheme" in href.lower() or len(title) > 15:
        # Avoid common header/navigation text
        if not any(skip in title.lower() for skip in ["home", "contact", "department", "back", "next"]):
            doc_content = f"Scheme Name: {title}. Managed by Agriculture and Farmers Welfare Department, Tamil Nadu Government."
            scheme_docs.append(Document(page_content=doc_content))

# Deduplicate scraped items
unique_docs = {doc.page_content: doc for doc in scheme_docs}.values()
scheme_docs = list(unique_docs)

print(f"Successfully scraped {len(scheme_docs)} schemes!")

if len(scheme_docs) == 0:
    print("Warning: Web scraping returned 0 items. Check internet connection or site access.")
    exit()

# 2. Extract Graph Entities using updated langchain_neo4j package
print("Processing Knowledge Graph Transformer...")
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Scheme", "Beneficiary", "Benefit", "Department"],
    allowed_relationships=["MANAGED_BY", "PROVIDES", "TARGETS"]
)

graph_documents = transformer.convert_to_graph_documents(scheme_docs)

# 3. Save directly to Neo4j
graph = Neo4jGraph()
graph.add_graph_documents(graph_documents)

print("Successfully ingested all scraped schemes into Neo4j!")