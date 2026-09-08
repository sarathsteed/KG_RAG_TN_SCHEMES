import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI

load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Refresh schema so LLM knows nodes/relationships
graph.refresh_schema()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)

def ask_chatbot(question: str):
    response = chain.invoke({"query": question})
    return response["result"]

if __name__ == "__main__":
    print("Tamil Nadu Farmers Welfare GraphRAG Chatbot Ready!\n")
    while True:
        user_input = input("Ask a question about TN Farmer Schemes (or 'exit'): ")
        if user_input.lower() == 'exit':
            break
        answer = ask_chatbot(user_input)
        print(f"\nChatbot: {answer}\n" + "-"*50)