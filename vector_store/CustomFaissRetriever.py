from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List

# Based on https://github.com/langchain-ai/langchain/issues/5067#issuecomment-1760834992
# Langchain currently does not support similarity scores when using a retriever (which i need for chain usage)
# it currently only supports db.similarity_search_with_score('xyz') but this is not a retriever classs

class CustomFaissRetriever(VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query, **self.search_kwargs
            )
        )

        if not docs_and_similarities:
            # No documents are found
            return []
        else:
            # Make the score part of the document metadata
            for doc, similarity in docs_and_similarities:
                doc.metadata["score"] = similarity

            docs = [doc for doc, _ in docs_and_similarities]
            return docs