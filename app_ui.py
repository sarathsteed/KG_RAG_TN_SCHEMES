import os
os.environ["XXHASH_FORCE_PURE_PYTHON"] = "1"

import streamlit as st
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="TN Farmers' Welfare Assistant", page_icon="🌾", layout="wide")
st.title("🌾 Tamil Nadu Farmers' Welfare Schemes Assistant")

load_dotenv()

@st.cache_resource
def init_resources():
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
    graph.refresh_schema()

    # Cypher generation prompt with explicit formatting rules
    CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
1. Use only the provided schema.
2. Nodes labeled `Scheme` have properties `id` and `name`.
3. If the user asks to "list" or "show" schemes, ALWAYS add `LIMIT 20`.
   Example: MATCH (s:Scheme) RETURN s.name AS scheme_name LIMIT 20
4. For text searches, use case-insensitive partial search:
   Example: MATCH (s:Scheme) WHERE toLower(s.name) CONTAINS toLower("nutrient") RETURN s.name AS scheme_name
5. Return scalar properties, NOT whole nodes. Do NOT use markdown code blocks.

Schema:
{schema}

Question: {question}
Cypher Query:"""

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"], 
        template=CYPHER_GENERATION_TEMPLATE
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )
    return chain, graph

try:
    chain, graph = init_resources()
except Exception as e:
    st.error(f"Connection error: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Tools
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Graph Tools")
    
    if st.button("📋 List First 20 Schemes (Direct Cypher)", use_container_width=True):
        raw_data = graph.query("MATCH (s:Scheme) RETURN s.name AS Scheme_Name LIMIT 20")
        st.write("**Database Records:**")
        st.dataframe(raw_data, use_container_width=True)

    if st.button("🔍 Search Micronutrient / Fertilizer Schemes", use_container_width=True):
        search_data = graph.query(
            "MATCH (s:Scheme) WHERE toLower(s.name) CONTAINS 'nutrient' OR toLower(s.name) CONTAINS 'fertilizer' OR toLower(s.name) CONTAINS 'spray' RETURN s.name AS Matching_Scheme"
        )
        st.write("**Matching Database Records:**")
        st.dataframe(search_data, use_container_width=True)

# ---------------------------------------------------------
# Custom Query Handler (Prevents "I don't know" for Lists)
# ---------------------------------------------------------
def query_assistant(user_input: str) -> str:
    lower_input = user_input.lower()
    
    # Direct Fallback 1: Catch "list" or "show all" queries
    if any(keyword in lower_input for keyword in ["list", "show all", "how many schemes", "all schemes"]):
        data = graph.query("MATCH (s:Scheme) RETURN s.name AS name LIMIT 20")
        if data:
            schemes_list = "\n".join([f"{idx+1}. {item['name']}" for idx, item in enumerate(data)])
            return f"Here are 20 schemes currently available in the Knowledge Graph:\n\n{schemes_list}"
        return "No schemes were found in the database."
    
    # Direct Fallback 2: Catch "micronutrient", "fertilizer", "spray" keyword queries
    if any(keyword in lower_input for keyword in ["micronutrient", "nutrient", "fertilizer", "spray"]):
        data = graph.query(
            "MATCH (s:Scheme) WHERE toLower(s.name) CONTAINS 'nutrient' OR toLower(s.name) CONTAINS 'fertilizer' OR toLower(s.name) CONTAINS 'spray' OR toLower(s.name) CONTAINS 'micro' RETURN s.name AS name"
        )
        if data:
            schemes_list = "\n".join([f"• {item['name']}" for item in data])
            return f"The following scheme(s) in the database relate to micronutrients or fertilizer assistance:\n\n{schemes_list}"
    
    # Standard Graph Cypher QA Chain execution
    res = chain.invoke({"query": user_input})
    return res.get("result", "No relevant details found in the graph.")

# ---------------------------------------------------------
# Chat UI
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask about TN farmer schemes...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching Knowledge Graph..."):
            try:
                response_text = query_assistant(user_query)
            except Exception as err:
                response_text = f"Query Error: `{str(err)}`"
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})