import os
os.environ["XXHASH_FORCE_PURE_PYTHON"] = "1"

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 1. Get Node Count
count = graph.query("MATCH (s:Scheme) RETURN count(s) AS total")[0]["total"]
print(f"Total Scheme Nodes in DB: {count}")

# 2. Fetch First 20 Scheme Names Directly via Cypher
schemes = graph.query("MATCH (s:Scheme) RETURN s.name AS name LIMIT 20")
print("\nFirst 20 Schemes in Database:")
for idx, s in enumerate(schemes, 1):
    print(f"{idx}. {s['name']}")