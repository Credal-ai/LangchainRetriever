from typing import List, Optional

import aiohttp
import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document


class CredalRetriever(BaseRetriever):
    """Credal.ai retriever"""

    base_url = "https://api.credal.ai"
    search_url = f"{base_url}/api/v0/search/searchDocumentCollection"
    document_collection_id: str
    metadata_filter_expression: Optional[str]
    max_chunks: Optional[int]
    merge_contents: Optional[bool]
    threshold: Optional[float]
    api_key: str

    def __search_options(self) -> dict:
        search_options = {}
        if self.max_chunks is not None:
            search_options["maxChunks"] = self.max_chunks
        if self.merge_contents is not None:
            search_options["mergeContents"] = self.merge_contents
        if self.threshold is not None:
            search_options["threshold"] = self.threshold
        return search_options

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        response = requests.post(
            self.search_url,
            json={
                "documentCollectionId": self.document_collection_id,
                "searchQuery": query,
                "metadataFilterExpression": self.metadata_filter_expression,
                "searchOptions": self.__search_options(),
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        results = response.json()
        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "document_id": result["documentId"],
                    "document_name": result["documentName"],
                    **result["documentMetadata"],
                    "chunk_id": chunk["chunkId"],
                    "chunk_index": chunk["chunkIndex"],
                    "chunk_score": chunk["score"],
                },
            )
            for result in results["results"]
            for chunk in result["chunks"]
        ]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                "POST",
                self.search_url,
                json={
                    "documentCollectionId": self.document_collection_id,
                    "searchQuery": query,
                    "metadataFilterExpression": self.metadata_filter_expression,
                    "searchOptions": self.__search_options(),
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            ) as response:
                results = await response.json()
        return [
            Document(
                page_content=chunk["text"],
                metadata={
                    "document_id": result["documentId"],
                    "document_name": result["documentName"],
                    **result["documentMetadata"],
                    "chunk_id": chunk["chunkId"],
                    "chunk_index": chunk["chunkIndex"],
                    "chunk_score": chunk["score"],
                },
            )
            for result in results["results"]
            for chunk in result["chunks"]
        ]
