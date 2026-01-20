from src.state.rag_state import RagState
class RagNodes:
    def __init__(self,retriever,llm):
        self.retriever=retriever
        self.llm=llm
    def retrieve_docs(self,state:RagState)->RagState:
        docs=self.retriever.invoke(state.question)
        return RagState(
            question=state.question,
            retrieved_docs=docs
        )
    def generate_answer(self,state:RagState)->RagState:
        context="\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt=f"Answer the question based on the context below:\nContext:{context}\nQuestion:{state.question}\nAnswer:"
        response=self.llm.invoke(prompt)
        return RagState(question=state.question,retrieved_docs=state.retrieved_docs,answer=response.content)