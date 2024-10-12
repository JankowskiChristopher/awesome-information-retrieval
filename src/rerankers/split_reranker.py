import logging
from typing import List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.dataset.beir.classes import RetrievalDataset
from src.evaluation.utils import merge_list_of_retrieval_results, split_dict_results
from src.rerankers.ensemble_reranker import EnsembleReranker
from src.utils import ResultDict

logger = logging.getLogger(__name__)


class SplitReranker(EnsembleReranker):
    """
    Class representing a reranker that splits the input results into multiple parts and reranks each part separately.
    When finished reranking, the results are merged back together.
    Splits mean which part of the retrieval results should be reranked by which reranker, however
    if this is an empty list, then the rerankers are applied in order and therefore the number of rerankers
    must be equal to the number of retrievers used in the ensemble.
    """

    def __init__(self, reranker_config: DictConfig, top_k: Optional[int] = None, reranker_type: str = "hybrid"):
        assert "splits" in reranker_config.keys(), "Splits must be defined in SplitReranker"
        self.splits = reranker_config.splits  # this might be an empty list
        self.top_k = top_k
        logger.info(f"Instantiating {reranker_type} SplitReranker with splits: {self.splits}")

        super().__init__(reranker_config)

    def rerank_results(self, results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Reranks the input results by splitting them into parts and reranking each part separately.
        Supports recursive reranking. The results after reranking are scaled and merged back together.

        :param results: Retrieved results to rerank
        :param dataset: Retrieval dataset
        :return: Reranked results in the same format as the input results
        """
        logger.info(f"Reranking results with SplitReranker. Type of results is {type(results)}.")
        if isinstance(results, list):
            logger.info(f"Number of results to rerank: {len(results)}")

        # Custom defined splits
        if self.splits:
            # Split the dataset into parts defined by splits.
            split_results: List[ResultDict] = []
            results_to_split = merge_list_of_retrieval_results(results)
            prev_split = 0  # previous split index, to calculate how many elements to take from the beginning
            for i, split in enumerate(self.splits):
                # split_dict_results expects how many elements to take from the beginning, not split index
                results_0, results_1 = split_dict_results(results_to_split, split - prev_split)
                split_results.append(results_0)
                # Add last split when we are at the end.
                if i == len(self.splits) - 1:
                    split_results.append(results_1)

                results_to_split = results_1
                prev_split = split
                logger.info(f"Split results into {len(split_results)} parts for splits: {self.splits}.")
        # No splits defined
        else:
            split_results = results if isinstance(results, list) else [results]
            logger.info("No splits defined. Reranking all results with rerankers in order.")

            assert len(self.subrerankers_configs) == len(split_results), (
                f"Number of rerankers must be equal to the number of retrievers used in the ensemble. "
                f"Got {len(self.subrerankers_configs)} rerankers and {len(split_results)} retrievers."
            )

        # Rerank each part separately.
        reranked_results = []
        for i, split_result in enumerate(split_results):
            # As rerankers are instantiated here, the memory will be saved.
            logger.info(f"Instantiating subreranker with config {self.subrerankers_configs[i]}.")
            with instantiate(self.subrerankers_configs[i]) as reranker:
                number_of_passages = max([len(v) for v in split_result.values()])
                reranker.set_automatic_batch_size(number_of_passages)
                logger.info(f"Reranking split {i} with reranker.")
                partial_results = reranker.rerank_results(split_result, dataset)
                assert len(partial_results) == 1, (
                    f"Expected only one result, got {len(partial_results)}."
                    f"Code currently does not support such operations."
                )
                top_k_passages = split_dict_results(partial_results[0], reranker.top_k)[0]  # take only top_k
                reranked_results.append(top_k_passages)

        # Merge the results back together.
        logger.info(f"Merging reranked results.")
        merged_results = merge_list_of_retrieval_results(reranked_results)
        # Len of results is guaranteed to be <= top_k as some retrievers e.g. elastic might return less results.

        return [merged_results]

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass
