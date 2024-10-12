# Code adapted from https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/search/lexical/bm25_search.py
import logging
import time
from typing import Dict, List, Optional

import tqdm
from datasets import Dataset

from src.dataset.beir.classes import CorpusType
from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.elastic_search import ElasticSearch

logger = logging.getLogger(__name__)


def sleep(seconds) -> None:
    """Sleep for a number of seconds."""
    if seconds > 0:
        time.sleep(seconds)


class ElasticSearchBM25Retriever(BaseRetriever):
    def __init__(
        self,
        hostname: str = "localhost",
        keys: Dict[str, str] = {"title": "title", "body": "txt"},
        language: str = "english",
        timeout: int = 100,
        retry_on_timeout: bool = True,
        maxsize: int = 24,
        num_workers: int = 1,  # Number of workers for the ElasticSearch client, keet it low to avoid overloading the server
        number_of_shards: int = "default",
        should_initialize: bool = True,
        sleep_for_refresh: int = 2,
        sleep_for_reconnect: int = 20,
        batch_size_retrieval: Optional[int] = 1,
        should_cache_results: bool = True,
        should_use_cached_results: bool = True,
        should_override_cache_with_less_docs: bool = False,
    ):
        super().__init__(should_cache_results, should_use_cached_results, should_override_cache_with_less_docs)
        logging.info("Initalizing ElasticLexicalRetriever")

        self.batch_size_retrieval = batch_size_retrieval
        self.docs_ids = None

        self.should_initialize = should_initialize
        self.sleep_for_refresh = sleep_for_refresh
        self.sleep_for_reconnect = sleep_for_reconnect
        self.num_workers = num_workers

        self.config = {
            "hostname": hostname,
            "keys": keys,
            "timeout": timeout,
            "retry_on_timeout": retry_on_timeout,
            "maxsize": maxsize,
            "number_of_shards": number_of_shards,
            "language": language,
        }
        self.es = None

    def get_name(self) -> str:
        return "ElasticLexicalRetriever"

    def initialize(self) -> None:
        """
        Method to initialize the ElasticSearch index.
        If the index already exists, it will be deleted and recreated.
        """
        self.es.delete_index()  # Delete index if it already exists
        sleep(self.sleep_for_refresh)  # Sleep to allow ElasticSearch to catch up with deleting index
        self.es.create_index()

    def index_corpus(self, corpus: CorpusType, dataset_name: str, dataset: Optional[Dataset] = None):
        self.config["index_name"] = dataset_name
        if not self.es:
            try:
                self.es = ElasticSearch(self.config)
            except Exception as e:
                logger.warning(f"Error connecting to ElasticSearch host={self.config['hostname']}: {e}")
                sleep(self.sleep_for_reconnect)
                self.es = ElasticSearch(self.config)

        if not self.should_initialize:
            return

        self.initialize()

        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {
            idx: {
                self.config["keys"]["title"]: corpus[idx].get("title", None),
                self.config["keys"]["body"]: corpus[idx].get("text", None),
            }
            for idx in list(corpus.keys())
        }

        start_time = time.perf_counter()
        logger.info("Starting indexing ElasticLexicalRetriever")

        self.es.bulk_add_to_index(
            generate_actions=self.es.generate_actions(dictionary=dictionary, update=False), progress=progress
        )
        logger.info(
            f"Finished indexing ElasticLexicalRetriever. Elapsed time: {time.perf_counter() - start_time} seconds"
        )
        sleep(self.sleep_for_refresh)  # Sleep to allow ElasticSearch to catch up with indexing

    def search(self, query: str, top_k: int) -> Dict[str, float]:
        # TODO: Implement search for a single query
        raise NotImplementedError("search for a single query is not implemented")

    def search_batch(self, query_ids: List[str], queries: List[str], top_k: int) -> List[Dict[str, float]]:
        es_results = self.es.lexical_multisearch(
            texts=queries, top_hits=top_k + 1
        )  # Add 1 extra if query is present with documents

        results = []
        for query_id, query, hit_results in zip(query_ids, queries, es_results):
            scores = {}
            for corpus_id, score in hit_results["hits"]:
                if corpus_id != query_id:  # Do not include the query itself in the results
                    scores[corpus_id] = score
            results.append(scores)
        return results

    def __exit__(self, *args, **kwargs):
        logger.info("__exit__ ElasticLexicalRetriever. Memory cleanup.")
        del self.es
        self.es = None
