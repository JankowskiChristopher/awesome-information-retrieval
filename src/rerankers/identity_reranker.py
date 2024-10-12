import logging
from typing import List

from evaluation.utils import merge_list_of_retrieval_results
from src.dataset.beir.classes import RetrievalDataset
from src.rerankers.base_reranker import BaseReranker
from src.utils import ResultDict

logger = logging.getLogger(__name__)


class IdentityReranker(BaseReranker):
    """
    Reranker that keeps the original order of the passages. Optionally filters the number of passages.
    """

    def __init__(self, top_k: int) -> None:
        logger.info(f"Creating identity reranker with top_k {top_k}.")
        self.top_k = top_k
        super().__init__()

    def rerank_results(self, results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Reranks the results from the retriever. The input format is a dict with the query id as a key and the value is
        also a dict with key being the doc_id and value being the score. Does not filter the results - it has to be done
        in other place to keep the interface the same between rerankers.

        :param results: Dict of queries and their respective retrieved passages and scores
        :param dataset: RetrievalDataset object
        :return: The same results.
        """
        results = merge_list_of_retrieval_results(results)
        return [results]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def set_automatic_batch_size(self, number_of_passages: int) -> None:
        pass
