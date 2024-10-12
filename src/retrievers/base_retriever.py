from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

from datasets import Dataset

from src.dataset.beir.classes import CorpusType


class BaseRetriever(ABC):
    """
    Abstract base class for implementing a retrieval system.
    This class provides the structure for indexing and searching documents or data.
    """

    should_cache_results: bool = True
    should_use_cached_results: bool = True
    should_override_cache_with_less_docs: bool = False
    batch_size_retrieval: Optional[int] = None

    def __init__(self, should_cache_results, should_use_cached_results, should_override_cache_with_less_docs):
        if should_override_cache_with_less_docs and not should_cache_results:
            raise ValueError(
                "should_override_cache_with_less_docs cannot be set to True if should_cache_results is set to False."
            )
        self.should_cache_results = should_cache_results
        self.should_use_cached_results = should_use_cached_results
        self.should_override_cache_with_less_docs = should_override_cache_with_less_docs

    @abstractmethod
    def index_corpus(self, corpus: CorpusType, dataset_name: str, dataset: Optional[Dataset] = None):
        """
        Indexes a corpus of documents or data for later retrieval.

        :param corpus: The corpus to index, typically a collection of documents or data.
        :param dataset_name: A unique identifier for the dataset being indexed.
        :param dataset: optional dataset object.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int) -> Dict[str, float]:
        """
        Searches the indexed corpus for the most relevant documents to a given query.

        :param query: The search query as a string.
        :param top_k: The number of top results to retrieve.
        :return: A dictionary of document IDs to their relevance scores.
        """
        pass

    @abstractmethod
    def search_batch(self, query_ids: List[str], queries: List[str], top_k: int) -> List[Dict[str, float]]:
        """
        Searches the indexed corpus for the most relevant documents for a batch of queries.

        :param query_ids: A list of unique identifiers for the queries.
        :param queries: A list of search queries as strings.
        :param top_k: The number of top results to retrieve for each query.
        :return: A list of dictionaries, each mapping document IDs to their relevance scores for each query.
        """
        pass

    def process_queries(self, query_ids: Iterable[str], queries: Iterable[str], batch_size: Optional[int] = None):
        """
        Processes a batch of queries and retrieves relevant documents.

        :param query_ids: An iterable of unique identifiers for the queries.
        :param queries: An iterable of search queries as strings.
        :param batch_size: The number of queries to process in a batch. If None, uses the class's default batch_size.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Returns a unique name for the retriever."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        """
        Subclasses free the resources used by the retriever here.
        """
        pass
