import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from datasets import Dataset
from llama_index import Document, ServiceContext, VectorStoreIndex
from llama_index.embeddings import BaseEmbedding
from llama_index.embeddings.base import SimilarityMode, similarity
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.simple import (
    LEARNER_MODES,
    MMR_MODE,
    SimpleVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
    _build_metadata_filter_fn,
    get_top_k_embeddings,
    get_top_k_embeddings_learner,
    get_top_k_mmr_embeddings,
)

from src.dataset.beir.classes import CorpusType
from src.retrievers.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


def get_query_method_with_similarity(similarity_mode: SimilarityMode = SimilarityMode.DOT_PRODUCT):
    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        similarity_fn = partial(similarity, mode=similarity_mode)

        """Get nodes for response."""
        # Prevent metadata filtering on stores that were persisted without metadata.
        if query.filters is not None and self._data.embedding_dict and not self._data.metadata_dict:
            raise ValueError(
                "Cannot filter stores that were persisted without metadata. "
                "Please rebuild the store with metadata to enable filtering."
            )
        # Prefilter nodes based on the query filter and node ID restrictions.
        query_filter_fn = _build_metadata_filter_fn(lambda node_id: self._data.metadata_dict[node_id], query.filters)

        if query.node_ids is not None:
            available_ids = set(query.node_ids)

            def node_filter_fn(node_id: str) -> bool:
                return node_id in available_ids

        else:

            def node_filter_fn(node_id: str) -> bool:
                return True

        node_ids = []
        embeddings = []
        # TODO: consolidate with get_query_text_embedding_similarities
        for node_id, embedding in self._data.embedding_dict.items():
            if node_filter_fn(node_id) and query_filter_fn(node_id):
                node_ids.append(node_id)
                embeddings.append(embedding)

        query_embedding = cast(List[float], query.query_embedding)

        if query.mode in LEARNER_MODES:
            top_similarities, top_ids = get_top_k_embeddings_learner(
                query_embedding,
                embeddings,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        elif query.mode == MMR_MODE:
            mmr_threshold = kwargs.get("mmr_threshold", None)
            top_similarities, top_ids = get_top_k_mmr_embeddings(
                query_embedding,
                embeddings,
                similarity_fn=similarity_fn,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
                mmr_threshold=mmr_threshold,
            )
        elif query.mode == VectorStoreQueryMode.DEFAULT:
            top_similarities, top_ids = get_top_k_embeddings(
                query_embedding,
                embeddings,
                similarity_fn=similarity_fn,
                similarity_top_k=query.similarity_top_k,
                embedding_ids=node_ids,
            )
        else:
            raise ValueError(f"Invalid query mode: {query.mode}")

        return VectorStoreQueryResult(similarities=top_similarities, ids=top_ids)

    return query


def index_path_factory(datasets_absolute_dir, dataset_name, index_filename):
    return Path(datasets_absolute_dir) / dataset_name / "indexes" / index_filename


class LlamaIndexRetriever(BaseRetriever):
    def __init__(
        self,
        context_embedding_model: BaseEmbedding,
        query_embedding_model: Optional[BaseEmbedding] = None,
        batch_size: int = 10,
        local: bool = False,
    ):
        super().__init__()
        self.corpus: Optional[CorpusType] = None
        self.dataset_name: Optional[str] = None
        self.dataset: Optional[Dataset] = None
        self.index_name: Optional[str] = None

        self.context_embedding_model = context_embedding_model
        self.query_embedding_model = query_embedding_model or context_embedding_model
        self.local = local
        self.retriever = None

    def index_corpus(self, corpus: CorpusType, dataset_name: str, dataset: Optional[Dataset] = None):
        self.corpus = corpus
        self.dataset_name = dataset_name
        self.dataset = dataset

        documents = []
        for id, val in corpus.items():
            doc = Document(text=val["text"], metadata={"title": val["title"], "doc_id": id})
            documents.append(doc)

        query_service_context = ServiceContext.from_defaults(embed_model=self.query_embedding_model, llm=None)
        context_service_context = ServiceContext.from_defaults(embed_model=self.context_embedding_model, llm=None)

        # Create index and retriever
        index = VectorStoreIndex.from_documents(documents, service_context=context_service_context, show_progress=True)
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=30)

        # Query encode is different from the context encoder and LlamaIndex does not support it by default yet
        self.retriever._service_context = query_service_context

        SimpleVectorStore.query = self.retriever.query = get_query_method_with_similarity()

    def search(self, query, top_k) -> Dict[str, float]:
        if self.retriever is None:
            raise RuntimeError("Index is not initialized. Please run index_corpus before using this method.")

        nodes_with_score = self.retriever.retrieve(query)
        results = {node.node.metadata["doc_id"]: node.score for node in nodes_with_score}

        return results
