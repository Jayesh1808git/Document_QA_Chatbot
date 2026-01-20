from langgraph.graph import StateGraph,END
from src.state.rag_state import RagState
from src.nodes.reactnode import RagNode
class GraphBuilder:
    def __init__(self,retriever,llm):
        self.nodes=RagNode(retriever,llm)
        self.graph=None
    def build(self):
        builder=StateGraph(RagState)
        builder.add_node("retriever",self.nodes.retrieve_docs)
        builder.add_node("responder",self.nodes.generate_answer)
        builder.set_entry_point("retriever")
        builder.add_edge("retriever","responder")
        builder.add_edge("responder",END)
        self.graph=builder.compile()
        return self.graph
    def run(self,question:str)->RagState:
        if self.graph is None:
            self.build()
        initial_state=RagState(question=question)
        return self.graph.invoke(initial_state)