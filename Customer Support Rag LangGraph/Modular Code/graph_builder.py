from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

def build_graph(retrieve_fn, generate_fn, GraphState):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve", RunnableLambda(retrieve_fn))
    graph.add_node("generate", RunnableLambda(generate_fn))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
