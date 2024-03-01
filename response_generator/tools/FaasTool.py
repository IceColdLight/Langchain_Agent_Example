from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from vector_store.Vector_Store import Vector_Store

class FaasInput(BaseModel):
    query: str = Field(description="search query to look up")

class FaasTool(BaseTool):
    name: str = "personal_data_retriever"
    description: str = (
        "A very accurate retriever that returns documents."
        "Only useful when you need to answer questions about the user's personal life."
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = FaasInput
    vector_store = Vector_Store()
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            docs = self.vector_store.get_retriever().get_relevant_documents(query)
            return self.format_docs(docs)
        except Exception as e:
            return repr(e)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
    
    def format_docs(self, docs):
        # This formats the docs from the vector db into one big string that is inserted into the prompt template of the LLM
        if(len(docs) == 0):
            return "No relevant documents found. Use your existing knowledge."
        else:
            output = "\n\n"
            for doc in docs:
                output += doc.page_content
            return output