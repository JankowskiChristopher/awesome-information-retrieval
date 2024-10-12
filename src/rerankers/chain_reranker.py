import logging
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig

from evaluation.utils import merge_list_of_retrieval_results
from src.dataset.beir.classes import RetrievalDataset
from src.rerankers.ensemble_reranker import EnsembleReranker
from src.utils import ResultDict

logger = logging.getLogger(__name__)


class ChainReranker(EnsembleReranker):
    """
    Class representing a chain reranker that is a composition of multiple rerankers.
    Rerankers are put into a pipeline where the output of the previous reranker is the input of the next reranker.
    """

    def __init__(self, reranker_config: DictConfig, reranker_type: str = "hybrid"):
        assert (
            "splits" not in reranker_config.keys()
        ), "Splits not allowed in ChainReranker"  # Splits used in SplitReranker. Just in case user forgets
        logger.info(f"Initializing {reranker_type} ChainReranker.")

        super().__init__(reranker_config)

    def rerank_results(self, results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Reranks the results using the rerankers in the pipeline.
        The output of the previous reranker is the input of the next reranker.

        :param results: Retrieval results
        :param dataset: Retrieval dataset
        :return: Reranked results in the same format as the input
        """
        logger.info(f"Reranking results with ChainReranker. Type of results is {type(results)}.")

        reranked_results: List[ResultDict] = results
        for i, reranker_config in enumerate(self.subrerankers_configs):
            logger.info(f"Chain reranker, instantiating reranker {i}.")
            with instantiate(reranker_config) as reranker:
                # rerank results, automatic batch size in set in EmbeddingReranker
                reranked_results = reranker.rerank_results(reranked_results, dataset)
        return reranked_results

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass
