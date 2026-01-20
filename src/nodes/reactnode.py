from typing import List
from src.state.rag_state import RagState
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools import Tool
class RagNode:
    def __init__(self,retriever,llm):
        self.retriever=retriever
        self.llm=llm
        self._agent=None
    def retrieve_docs(self,state:RagState)->RagState:
        docs=self.retriever.invoke(state.question)
        return RagState(
            question=state.question,
            retrieved_docs=docs
        )
    def build_tools(self)->List[Tool]:
        def retriever_tool_fn(query:str)->str:
            docs:List[Document]=self.retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            merge=[]
            for i,d in enumerate(docs,start=1):
                meta=d.metadata if hasattr(d,"metadata") else {}
                title=meta.get("title",f"Document {i}")
                merge.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merge)
        retriever_tool=Tool(
            name="retriever",
            description="Fetches relevant docs in the database",
            func=retriever_tool_fn
        )
        wiki=WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3,lang="en")
        )
        wikipedia_tool=Tool(
            name="wikipedia",
            description="Searches wikipedia if the answer is not in vector store",
            func=wiki.run,
        )
        return [retriever_tool,wikipedia_tool]
    def _build_agent(self):
        pass
    
    def generate_answer(self,state:RagState)->RagState:
        context="\n".join([doc.page_content for doc in state.retrieved_docs])
        system_prompt="""You are a Contract Analysis Specialist. Your goal is to extract, summarize, and flag risks in legal agreements.

Task: Analyze the provided contract excerpts to answer the user's query.

Review Guidelines:

Risk Identification: If a clause is unusually one-sided or missing standard protections (e.g., a mutual indemnity), flag it as a "Potential Risk."

Cross-Referencing: Check if a term used in a clause is defined in the "Definitions" section of the provided context.

No Hallucination: If the contract is silent on a topic, state: "The document does not address [topic]."

Disclaimer: Conclude every response with: "This analysis is for informational purposes only and does not constitute legal advice."""
        
        prompt=f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {state.question}\n\nAnswer:"
        response=self.llm.invoke(prompt)
        answer=response.content if hasattr(response,"content") else str(response)
        
        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )
