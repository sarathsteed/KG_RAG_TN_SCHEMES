import os
import json
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

load_dotenv()

# Initialize Neo4j Connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Initialize LLM & Graph Transformer
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Scheme", "Beneficiary", "Benefit", "Requirement", "Department"],
    allowed_relationships=["PROVIDES", "ELIGIBLE_FOR", "REQUIRES", "MANAGED_BY"]
)

# Load Scraped Schemes
with open("schemes.json", "r") as f:
    schemes_data = json.load(f)

# Convert items to Documents and extract graph entities
docs = [Document(page_content=f"Scheme Title: {item['title']}") for item in schemes_data]
graph_documents = transformer.convert_to_graph_documents(docs)

# Store in Neo4j
graph.add_graph_documents(graph_documents)
print("Graph built successfully in Neo4j!")