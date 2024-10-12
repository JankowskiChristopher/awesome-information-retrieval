import gc
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import torch

from src.constants import MODELS_ROOT_DIR, NAS_DIR
from src.dataset.beir.classes import RetrievalDataset
from src.evaluation.utils import (
    convert_dict_results_to_llmrank_list_results,
    convert_llmrank_list_results_to_dict_results,
    inform_about_copy_error,
    merge_list_of_retrieval_results,
)
from src.rerankers.base_reranker import BaseReranker
from src.rerankers.rank_llm.result import LLMRerankerResult
from src.rerankers.rank_llm.zephyr_reranker import ZephyrReranker
from src.utils import ResultDict

logger = logging.getLogger(__name__)


class LLMReranker(BaseReranker):
    """
    Subclass of the BaseReranker. Reranks the results from the retriever using an LLM reranking model.
    By default, the model is RankZephyr, but in the future maybe will be different.
    Thanks to using an LLM, the attention mechanism spans over all passages and the query.
    """

    def __init__(self, top_k: Optional[int] = None):
        start_time = time.perf_counter()

        hf_model_cache_dir = Path("/repo/hf/models")
        nas_models_dir = Path(NAS_DIR) / MODELS_ROOT_DIR
        model_name_on_disk = "models--castorini--rank_zephyr_7b_v1_full"
        locks_disk_dir = "rank_zephyr_locks"
        save_locally = False
        self.top_k = top_k  # number of passages to return from the reranker

        # Check if the model is available
        if not (hf_model_cache_dir / model_name_on_disk).exists():
            logger.info(f"RankZephyr not found in {str(hf_model_cache_dir)}. Trying to copy from /nas")
            # Copy the model from /nas
            if (Path(nas_models_dir) / model_name_on_disk).exists() and (
                Path(nas_models_dir) / locks_disk_dir
            ).exists():
                logger.info(f"Copying RankZephyr from {str(nas_models_dir)} to {str(hf_model_cache_dir)}")
                exit_code = os.system(f"cp -r {str(nas_models_dir / model_name_on_disk)} {str(hf_model_cache_dir)}")
                inform_about_copy_error(exit_code)
                # Copy locks. Probably need to create .locks directory first. Copy all locks from /nas.
                exit_code = os.system(
                    f"mkdir -p {str(hf_model_cache_dir / '.locks' / model_name_on_disk)} && "
                    f"cp {str(nas_models_dir / locks_disk_dir / '*')} {str(hf_model_cache_dir / '.locks' / model_name_on_disk)}"
                )
                inform_about_copy_error(exit_code)
            else:
                logger.info(
                    f"RankZephyr not found in {str(nas_models_dir)}. " f"Model will be downloaded from HuggingFace."
                )
                save_locally = True

        self.model = ZephyrReranker()
        if save_locally:
            logger.info(f"Downloaded RankZephyr, copying to {str(nas_models_dir)}")
            exit_code = os.system(f"cp -r {str(hf_model_cache_dir / model_name_on_disk)} {str(nas_models_dir)}")
            inform_about_copy_error(exit_code)
            exit_code = os.system(
                f"cp -r {str(hf_model_cache_dir / '.locks' / model_name_on_disk)} {str(nas_models_dir / locks_disk_dir)}"
            )
            inform_about_copy_error(exit_code)

        logger.info(f"Finished loading LLM reranker. Elapsed time: {time.perf_counter() - start_time} seconds.")

    @torch.inference_mode()
    def rerank_results(self, retriever_results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Reranks the results from the retriever. The input format is a dict with the query id as a key and the value is
        also a dict with key being the doc_id and value being the score. Thanks to batching the performance is better.

        :param retriever_results: Dict of queries and their respective retrieved passages and scores
        :param dataset: RetrievalDataset object
        :return: New results in the same format as the input.
        """
        start_time = time.perf_counter()
        logger.info("Start reranking results with LLM Reranker.")

        retriever_results = merge_list_of_retrieval_results(retriever_results)
        reranker_input_list: List[LLMRerankerResult] = convert_dict_results_to_llmrank_list_results(
            retriever_results, dataset
        )

        logger.info(f"Reranking {len(reranker_input_list)} queries of type List[Result].")
        rerank_results: List[LLMRerankerResult] = self.model.rerank(reranker_input_list)
        # Filter results based on rank. Keep only top_k if top_k is not None,
        if self.top_k is not None:
            for rerank_result in rerank_results:
                rerank_result.hits = [
                    hit for hit in rerank_result.hits if hit["rank"] <= self.top_k
                ]  # ranks start with 1
        new_results = convert_llmrank_list_results_to_dict_results(rerank_results)

        logger.info(f"Finished LLM reranking. Elapsed time: {time.perf_counter() - start_time} seconds.")

        return [new_results]

    def set_automatic_batch_size(self, number_of_passages: int) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.model:
            del self.model
            self.model = None

        gc.collect()
        torch.cuda.empty_cache()
