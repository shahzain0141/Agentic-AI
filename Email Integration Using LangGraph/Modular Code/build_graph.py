from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from graph_nodes import GraphState, get_retrieve_node, generate, send_email_node

def build_graph(db):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", get_retrieve_node(db))
    graph.add_node("generate", RunnableLambda(generate))
    graph.add_node("send_email", RunnableLambda(send_email_node))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "send_email")
    graph.add_edge("send_email", END)
    
    return graph.compile()
