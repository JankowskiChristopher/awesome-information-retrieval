import logging
import time
from typing import Dict, List, Optional

from datasets import Dataset
from rank_bm25 import BM25Okapi

from src.dataset.beir.classes import CorpusType
from src.retrievers.base_retriever import BaseRetriever

_tokenizer = lambda x: x.split(" ")

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        num_workers: int = 2,
        batch_size_retrieval: Optional[int] = 1,
        should_cache_results: bool = True,
        should_use_cached_results: bool = True,
        should_override_cache_with_less_docs: bool = False,
    ):
        super().__init__(should_cache_results, should_use_cached_results, should_override_cache_with_less_docs)
        logging.info("Initalizing B25 retriever")

        self.num_workers = num_workers
        self.batch_size_retrieval = batch_size_retrieval
        self.docs_ids = None
        self.bm25 = None

    def get_name(self) -> str:
        return "BM25Retriever"

    def index_corpus(self, corpus: CorpusType, dataset_name: str, dataset: Optional[Dataset] = None):
        self.docs_ids = list(corpus.keys())
        tokenized_corpus = [_tokenizer(doc["text"]) for doc in corpus.values()]

        start_time = time.perf_counter()
        logger.info("Starting indexing BM25")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Finished indexing BM25. Elapsed time: {time.perf_counter() - start_time} seconds")

    def search(self, query: str, top_k: int) -> Dict[str, float]:
        tokenized_query = _tokenizer(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_docs = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        return {self.docs_ids[i]: doc_scores[i] for i in top_docs}

    def search_batch(self, query_ids: List[str], queries: List[str], top_k: int) -> List[Dict[str, float]]:
        results = []
        for query in queries:
            result = self.search(query, top_k)
            results.append(result)
        return results

    def __exit__(self, *args, **kwargs):
        logger.info("__exit__ BM25Retriever. Memory cleanup.")
        del self.bm25
        self.bm25 = None
