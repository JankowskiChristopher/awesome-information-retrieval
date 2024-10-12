import gc
import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from src.dataset.beir.classes import RetrievalDataset
from src.evaluation.utils import convert_dict_results_to_lists, merge_list_of_retrieval_results
from src.rerankers.base_reranker import BaseReranker
from src.utils import (
    ResultDict,
    assert_len_list_equal,
    batch_list,
    check_total_gpu_memory,
    flatten_list,
    get_model,
    get_tokenizer,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EmbeddingReranker(BaseReranker):
    """
    Subclass of the BaseReranker. Reranks the results from the retriever using the embedding model which is
    based on the Roberta model - it takes both the query and the passage as input and outputs a score attending
    over all the tokens in the query and the passage. Used model is BAAI/bge-reranker-base or large.
    As the model jointly attends over input, the embeddings of the passages cannot be stored on the disk.
    The large model has about 3x more parameters than the base model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        tokenizer_name: str = "BAAI/bge-reranker-large",
        top_k: Optional[int] = None,
        local: bool = False,
        batch_size: int = 10,
        device: str = "cuda",
    ) -> None:
        logger.info(f"Creating reranker with model {model_name}.")
        self.model = get_model(model_name, device=device, local=local, auto_model_classification=True)
        self.tokenizer = get_tokenizer(tokenizer_name, local=local)
        self.top_k = top_k  # If not None or 0, then limit the number of reranked passages to top k.
        self.batch_size = batch_size
        self.device = device
        super().__init__()

    def rerank_results(self, results: List[ResultDict], dataset: RetrievalDataset) -> List[ResultDict]:
        """
        Reranks the results from the retriever. The input format is a dict with the query id as a key and the value is
        also a dict with key being the doc_id and value being the score. Thanks to batching the performance is better.

        :param results: Dict of queries and their respective retrieved passages and scores
        :param dataset: RetrievalDataset object
        :return: New results in the same format as the input.
        """
        start_time = time.perf_counter()
        logger.info(f"Start reranking results with EmbeddingReranker and batch_size {self.batch_size}.")

        results = merge_list_of_retrieval_results(results)
        # set automatic batch size
        self.set_automatic_batch_size(max([len(v) for v in results.values()]))

        v = convert_dict_results_to_lists(results, dataset)
        query_ids_list, queries_list, doc_ids_list, passages_list = v.query_ids, v.queries, v.doc_ids, v.passages

        batched_query_ids = batch_list(query_ids_list, self.batch_size)
        batched_doc_ids = batch_list(doc_ids_list, self.batch_size)
        batched_queries = batch_list(queries_list, self.batch_size)
        batched_passages = batch_list(passages_list, self.batch_size)

        new_results = {}
        logger.info(f"{len(batched_query_ids)} batches to rerank.")

        for b in tqdm(range(len(batched_query_ids))):
            logger.debug(
                f"len of batched_queries {len(batched_queries[b])} "
                f"and len of batched_passages {len(batched_passages[b])}."
            )
            batched_new_scores, batched_new_indices = self._rerank(batched_queries[b], batched_passages[b])

            assert_len_list_equal(batched_queries[b], batched_new_scores)
            assert_len_list_equal(batched_passages[b], batched_new_indices)

            for query_id, doc_ids, new_scores, new_indices in zip(
                batched_query_ids[b], batched_doc_ids[b], batched_new_scores, batched_new_indices
            ):
                results_value: Dict[str, float] = {}
                sorted_doc_ids = [doc_ids[i] for i in new_indices]

                assert_len_list_equal(sorted_doc_ids, new_scores)

                for doc, score in zip(sorted_doc_ids, new_scores):
                    results_value[doc] = score.item()

                new_results[query_id] = results_value

        logger.info(f"Finished reranking. Elapsed time: {time.perf_counter() - start_time} seconds.")

        return [new_results]

    def _rerank(self, queries: List[str], docs: List[List[str]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Reranks the docs based on the query. Due to ensemble retrievers, docs do not have a fixed size for each
        element in the list as duplicates are removes. Therefore, we operate on lists instead of tensors.
        Returns the new scores and new indices of the reranked docs and queries.

        :param queries: Queries for which passages were retrieved
        :param docs: List of passages retrieved to be reranked
        :return: Tuple of sorted passages and their respective scores
        """
        self.model.eval()
        # Expand number of queries to match the number of docs and keep track of the lens (may be different in batches).
        list_of_queries: List[List[str]] = [[q] * len(docs[i]) for i, q in enumerate(queries)]
        list_of_queries_lens = [len(lst) for lst in list_of_queries]

        # debug and test
        logger.debug(f"list_of_queries lens {[len(lst) for lst in list_of_queries]}")
        logger.debug(f"docs lens {[len(lst) for lst in docs]}")
        assert_len_list_equal(flatten_list(list_of_queries), flatten_list(docs))

        # Embedding reranker expects pairs (query, doc)
        pairs = [[p, d] for p, d in zip(flatten_list(list_of_queries), flatten_list(docs))]
        logger.debug(f"pairs lens {len(pairs)}")

        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = (
                self.model(**inputs.to(self.device), return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        # Collect results and put the docs back to corresponding queries.
        start_batch_index = 0
        list_scores: List[torch.Tensor] = []
        for number_of_queries_in_batch in list_of_queries_lens:
            list_scores.append(scores[start_batch_index : start_batch_index + number_of_queries_in_batch])
            start_batch_index += number_of_queries_in_batch

        # test shapes
        for i, s in enumerate(list_scores):
            assert s.shape[0] == list_of_queries_lens[i], (
                f"Shapes of scores and queries do not match. " f"Got {s.shape[0]} and {list_of_queries_lens[i]}."
            )

        # We operate on lists of tensors as batches are not guaranteed to be of the same size.
        sorted_scores_list = []
        sorted_indices_list = []
        for scores in list_scores:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            sorted_scores_list.append(sorted_scores)
            sorted_indices_list.append(sorted_indices)

        # Reranker can limit the number of reranked passages.
        if (self.top_k is not None) and (self.top_k > 0):
            sorted_scores_list = [sorted_score[: self.top_k] for sorted_score in sorted_scores_list]
            sorted_indices_list = [sorted_index[: self.top_k] for sorted_index in sorted_indices_list]
            return sorted_scores_list, sorted_indices_list

        logger.debug(f"Lens of sorted scores and indices: {len(sorted_scores_list)} and {len(sorted_indices_list)}.")
        return sorted_scores_list, sorted_indices_list

    def set_automatic_batch_size(self, number_of_passages: int):
        """
        FUnction to automatically set the batch size based on the number of passages to rerank.
        Uses a heuristic tested on BEIR benchmark.
        :param number_of_passages:
        :return:
        """
        gpu_mem_gb = check_total_gpu_memory()
        magic_heuristic_number = 13.24  # determined through testing. Almost optimal and avoids out of memory errors
        old_batch_size = self.batch_size
        self.batch_size = int(gpu_mem_gb * magic_heuristic_number) // number_of_passages
        if self.batch_size < 1:
            logger.warning(
                f"Batch size is less than 1. Setting it to 1. Old batch size: {old_batch_size}, "
                f"new batch size: {self.batch_size}."
            )
            self.batch_size = 1
        if self.batch_size != old_batch_size:
            logger.info(
                f"Setting automatically batch size to {self.batch_size}."
                f"GPU memory is {gpu_mem_gb} GiB."
                f"Number of passages to rerank is {number_of_passages}."
            )

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if self.model:
            del self.model
            self.model = None

        gc.collect()
        torch.cuda.empty_cache()
