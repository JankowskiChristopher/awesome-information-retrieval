import concurrent
import gc
import logging
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Callable, Dict, List, Optional

import torch
import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
from hydra.utils import instantiate
from loky import ProcessPoolExecutor
from omegaconf import DictConfig, omegaconf

from src.constants import CQADUPSTACK_SUBFORUMS
from src.dataset.beir.classes import RetrievalDataset
from src.dataset.beir.utils import get_dataset
from src.evaluation.utils import MetricDict, merge_list_of_retrieval_results
from src.rerankers.base_reranker import BaseReranker
from src.retrievers.base_retriever import BaseRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.elastic_search_bm25_retriever import ElasticSearchBM25Retriever
from src.retrievers.embedding_retriever import EmbeddingRetriever
from src.retrievers.ensemble_retriever import EnsembleRetriever
from src.utils import (
    ResultDict,
    assert_dict_contained,
    assert_dict_num_passages,
    assert_len_list_equal,
    batch_iterable,
    load_dict_results,
    save_dict_results,
)
from src.writer import Writer

logger = logging.getLogger(__name__)


class BeirEvaluator:
    def __init__(
        self,
        cfg: DictConfig,
        writer: Writer,
        retriever: BaseRetriever,
        instantiate_reranker_func: Callable[[], Optional[BaseReranker]] = None,
        split: str = "test",
    ) -> None:
        self.local = cfg.local
        self.retriever = retriever
        self.instantiate_reranker_func = instantiate_reranker_func
        self.writer = writer
        self.split = split
        self.cfg = cfg

    def run(self, datasets_names: List[str], metrics_k_values: List[int]) -> None:
        for dataset_name in datasets_names:
            try:
                start_time = time.perf_counter()
                self.evaluate_dataset(dataset_name, metrics_k_values)
                execution_time = time.perf_counter() - start_time
                logger.info(f"Execution time for {dataset_name}: {execution_time} seconds")
                # Clear memory between datasets just in case,
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(
                    f"Error while evaluating dataset {dataset_name}. Error: {e}\nstacktrace {traceback.print_exc()}"
                )

    def evaluate_dataset(self, dataset_name: str, metrics_k_values: List[int]):
        logger.info(f"Evaluating on dataset: {dataset_name}")
        logger.info("-------------------------------------")

        split = "dev" if dataset_name == "msmarco" else self.split  # Use dev on MSMarco, as test contains only 43 qrels

        # Special case for CQADupStack - evaluate on all subforums
        if dataset_name == "cqadupstack":
            subforums_metrics: MetricDict = {}
            for subforum in CQADUPSTACK_SUBFORUMS:
                subforum_dataset_name = f"cqadupstack_{subforum}"
                subforum_dataset = get_dataset(subforum_dataset_name, split=split, local=self.local)
                subforum_metrics = self.evaluate_on_single_dataset(
                    subforum_dataset, subforum_dataset_name, metrics_k_values
                )
                subforums_metrics[subforum] = subforum_metrics

            # Aggregate results from all subforums - for each metric take the unweighted mean of all subforums
            final_metrics = self.aggregate_metrics_from_datasets(subforums_metrics)
        else:
            dataset = get_dataset(dataset_name, split=split, local=self.local)
            final_metrics = self.evaluate_on_single_dataset(dataset, dataset_name, metrics_k_values)

        self.log_evaluation_results(final_metrics, dataset_name)

    def evaluate_on_single_dataset(
        self, dataset: RetrievalDataset, dataset_name: str, metrics_k_values: List[int]
    ) -> MetricDict:
        """
        Evaluate the retriever from the config on a single dataset.
        The function retrieves the results and reranks them if configured.

        :param dataset: The dataset to evaluate on.
        :param dataset_name: The name of the dataset.
        :param metrics_k_values: The values of k for which to calculate the metrics.

        :return: The metrics for the dataset.
        The metrics are a dictionary with the metric name as key and the metric value as value.
        """

        def _retrieve(
            _retriever: BaseRetriever, num_passages_to_retrieve: int, _dataset: RetrievalDataset, _dataset_name: str
        ) -> ResultDict:
            """
            Helper function to avoid deduplication of code when dealing with the ensemble reranker.
            :param _retriever: The retriever to use.
            :param num_passages_to_retrieve: The number of passages to retrieve.
            :param _dataset: The dataset to retrieve on.
            :param _dataset_name: The name of the dataset.

            :return: The retrieved results.
            """
            # Arguana is a special case with out of memory errors.
            if _dataset_name == "arguana" and isinstance(_retriever, EmbeddingRetriever):
                _retriever.batch_size_passages //= 8
                _retriever.batch_size_queries //= 8
            else:
                if hasattr(self.cfg.retriever, "batch_size_passages"):
                    _retriever.batch_size_passages = self.cfg.retriever.batch_size_passages
                if hasattr(self.cfg.retriever, "batch_size_queries"):
                    _retriever.batch_size_queries = self.cfg.retriever.batch_size_queries

            _original_retriever_results = None
            if _retriever.should_use_cached_results:
                _original_retriever_results = load_dict_results(
                    model_name=_retriever.get_name(),
                    dataset_name=_dataset_name,
                    top_k=num_passages_to_retrieve,
                    local=self.local,
                )

            # If results could not be loaded from cache
            # (because they do not exist or the number of saved passages is not sufficient), retrieve them.
            if _original_retriever_results is None:
                _retriever.index_corpus(corpus=_dataset.corpus, dataset_name=_dataset_name)
                _original_retriever_results = self.retrieve_results(_retriever, _dataset, num_passages_to_retrieve)

            if _retriever.should_cache_results:
                save_dict_results(
                    _original_retriever_results,
                    model_name=_retriever.get_name(),
                    dataset_name=_dataset_name,
                    should_override_cache_with_less_docs=_retriever.should_override_cache_with_less_docs,
                    num_passages=num_passages_to_retrieve,
                    local=self.local,
                )

            return _original_retriever_results

        # Retrieval
        num_passages_to_retrieve = 0
        list_of_results = []  # single retriever and ensemble retriever both store results in a list

        # Separate case when the retriever is an ensemble
        if isinstance(self.retriever, EnsembleRetriever):
            for retriever_config_path, _num_passages_to_retrieve in zip(
                self.retriever.configs_paths, self.retriever.nums_of_retrieved_passages
            ):
                logger.info(f"Ensemble retriever: {retriever_config_path}. Num passages {_num_passages_to_retrieve}")
                retriever = instantiate(omegaconf.OmegaConf.load(retriever_config_path))
                with retriever as retriever:
                    list_of_results.append(_retrieve(retriever, _num_passages_to_retrieve, dataset, dataset_name))
                num_passages_to_retrieve += _num_passages_to_retrieve

            assert len(list_of_results) >= 2, "Ensemble retriever should have at least 2 retrievers."

            # Merging is done later if no reranker was provided
            logger.info("Finished retrieving with ensemble retriever.")
        # Normal case
        else:
            with self.retriever as retriever:
                _num_passages_to_retrieve = max(self.cfg.num_retrieved_passages, max(metrics_k_values))
                list_of_results = [_retrieve(retriever, _num_passages_to_retrieve, dataset, dataset_name)]
                num_passages_to_retrieve = _num_passages_to_retrieve
                logger.info("Finished retrieving with single retriever.")

        # We potentially need to merge the results of the ensemble retriever. If no reranker is provided,
        # these will be the final results, otherwise they are still necessary for the sanity checks.
        original_retriever_results = merge_list_of_retrieval_results(list_of_results)

        # Reranking
        if self.instantiate_reranker_func is not None:
            logger.info("Starting reranking")
            with self.instantiate_reranker_func() as reranker:
                # Set the automatic batch size to the number of passages to retrieve
                # In some rerankers this function has no behaviour.
                reranker.set_automatic_batch_size(num_passages_to_retrieve)
                logger.info("Reranking all passages.")
                results = reranker.rerank_results(list_of_results, dataset)
        # No reranking
        else:
            logger.info("No reranker provided. Using original retriever results.")
            results = original_retriever_results

        # Convert results to a single dictionary
        if isinstance(results, list):
            logger.info("Converting results to a single ResultDict before evaluation.")
            assert len(results) == 1, "Results should be a single dictionary."
            results = results[0]

        self.run_sanity_checks(original_retriever_results, results, dataset)
        metrics = self.evaluate_results(dataset, results, metrics_k_values)

        return metrics

    def run_sanity_checks(
        self, original_retriever_results: ResultDict, results: ResultDict, dataset: RetrievalDataset
    ) -> None:
        """
        Function runs sanity checks to ensure that the results of the retriever and the reranker are correct:
        - The results are always a subset (or equal) of the original retriever results.
        - The number of queries is the same after retrieval and reranking.
        - The number of passages is correct after retrieval and reranking.

        :param original_retriever_results: The results of the retriever.
        :param results: The results of the reranker.
        If no reranker is used, these should be the same as the original retriever results.
        :param dataset: The dataset used for retrieval and reranking.

        :return: None
        """

        assert_dict_contained(original_retriever_results, results)  # results are always a subset (or equal).
        # Additional check whether the number of queries is the same after retrieval and reranking
        assert_len_list_equal(list(dataset.queries.keys()), list(original_retriever_results.keys()))
        assert_len_list_equal(list(original_retriever_results.keys()), list(results.keys()))
        # Check if the number of passages is correct
        if self.instantiate_reranker_func is not None:  # If reranker is used, check the exact number of passages
            comparison_func = lambda passage_count, top_k: top_k == passage_count
        else:  # If no reranker is used, check if the number of passages is at least the number of passages to retrieve
            comparison_func = lambda passage_count, top_k: top_k <= passage_count

        if not isinstance(
            self.retriever, ElasticSearchBM25Retriever
        ):  # ElasticSearchBM25Retriever can return fewer passages
            assert_dict_num_passages(results, max(self.cfg.metrics_k_values), comparison_func)

    def retrieve_information(self, retriever: BaseRetriever, query_ids, queries, batch_size, num_cores, max_k):
        """
        Retrieve information for the given queries using a batched approach with parallel processing.

        :param retriever: The retriever to use.
        :param query_ids: A list of all query IDs.
        :param queries: A list of all queries.
        :param batch_size: The size of each batch.
        :param num_cores: The number of cores to use for parallel processing.
        :param max_k: The maximum number of results to retrieve for each query.

        :return: A dictionary of results with query IDs as keys and results as values.
        """

        def process_batch(query_ids_batch, queries_batch):
            results_batch = retriever.search_batch(query_ids_batch, queries_batch, top_k=max_k)
            return [(_query_id, _result) for _query_id, _result in zip(query_ids_batch, results_batch)]

        num_workers = retriever.num_workers if hasattr(retriever, "num_workers") else num_cores
        executor_class = ProcessPoolExecutor if isinstance(retriever, BM25Retriever) else ThreadPoolExecutor

        if executor_class is ProcessPoolExecutor:
            logger.info("Using ProcessPoolExecutor")
            num_workers = retriever.num_workers
            batch_size = ceil(len(query_ids) / num_workers)

        logger.info(f"Started retrieving results with batch_size={batch_size}, num_workers={num_workers}")

        batched_query_ids = batch_iterable(query_ids, batch_size)
        batched_queries = batch_iterable(queries, batch_size)

        results = {}

        start_time = time.perf_counter()
        with executor_class(max_workers=num_workers) as executor:
            futures = (
                executor.submit(process_batch, query_ids, queries)
                for query_ids, queries in zip(batched_query_ids, batched_queries)
            )

            num_iters = ceil(len(query_ids) / batch_size)

            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=num_iters):
                batch_results = future.result()
                for query_id, result in batch_results:
                    results[query_id] = result

        logger.info(
            f"Finished retrieving results with max_workers={num_workers}, batch_size={batch_size}. "
            f"Elapsed time: {time.perf_counter() - start_time} seconds"
        )

        return results

    def retrieve_results(self, retriever: BaseRetriever, dataset: RetrievalDataset, max_k: int) -> ResultDict:
        start_time = time.perf_counter()
        logger.info("Evaluating retriever on questions against qrels")

        if retriever.batch_size_retrieval:
            query_ids = dataset.queries.keys()
            queries = dataset.queries.values()

            retriever.process_queries(query_ids, queries)

            logger.info(f"Finished embedding queries. Elapsed time: {time.perf_counter() - start_time} seconds")
            start_time = time.perf_counter()

            batch_size = retriever.batch_size_retrieval
            num_cores = os.cpu_count()

            results = self.retrieve_information(retriever, query_ids, queries, batch_size, num_cores, max_k)
        else:
            results = {}
            for key, query in tqdm.tqdm(dataset.queries.items()):
                results[key] = retriever.search(query, top_k=max_k)

        logger.info(f"Finished retrieving results. Elapsed time: {time.perf_counter() - start_time} seconds")
        return results

    def evaluate_results(
        self,
        dataset: RetrievalDataset,
        results: ResultDict,
        metrics_k_values: List[int],
    ) -> MetricDict:
        """
        Evaluate the retrieval results on the dataset. The function calculates the metrics for the given k values.
        Returns a dictionary with the metric name as key and the metric value as value.

        :param dataset: The dataset to evaluate on.
        :param results: The results of the retrieval.
        :param metrics_k_values: The values of k for which to calculate the metrics.

        :return: The metrics for the dataset. The metrics are a dictionary with the metric name as key and the metric
        value as value.
        """

        ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(dataset.qrels, results, metrics_k_values)

        all_metrics: MetricDict = {}
        for k in metrics_k_values:
            metrics = {
                f"NDCG@{k}": ndcg[f"NDCG@{k}"],
                f"MAP@{k}": map_[f"MAP@{k}"],
                f"Recall@{k}": recall[f"Recall@{k}"],
                f"precision@{k}": precision[f"P@{k}"],
            }
            all_metrics.update(metrics)

        return all_metrics

    def aggregate_metrics_from_datasets(
        self,
        dict_of_metrics: Dict[str, MetricDict],
        aggregate_function: Optional[Callable[[List[float]], float]] = None,
    ) -> MetricDict:
        """
        Aggregate metrics from multiple datasets into a single result dictionary using a provided aggregation function.

        :param dict_of_metrics: A dictionary where the key is the dataset name and the value
                is another dictionary containing metric data.
        :param aggregate_function: A function that takes a list of metric
                values and returns an aggregated result. Defaults to mean if not provided.

        :return: A dictionary with the aggregated metric values.
        The keys are the metric names and the values are the aggregated metric values.

        Raises:
            ValueError: If no datasets are provided for aggregation.
        """

        num_datasets = len(dict_of_metrics)
        if num_datasets == 0:
            raise ValueError("No datasets to aggregate.")

        if aggregate_function is None:

            def aggregate_function(results_list: List[float]) -> float:
                if not results_list:
                    raise ValueError("No results to aggregate.")
                return sum(results_list) / len(results_list)

        metrics_dict: Dict[str, List[float]] = defaultdict(
            list
        )  # Initialize a dictionary with empty lists for each metric
        for dataset_name, metrics in dict_of_metrics.items():
            for metric_name, metric_value in metrics.items():
                metrics_dict[metric_name].append(metric_value)

        aggregated_metrics = {
            metric_name: aggregate_function(results_list) for metric_name, results_list in metrics_dict.items()
        }

        return aggregated_metrics

    def log_evaluation_results(self, metrics: MetricDict, dataset_name: str) -> None:
        """
        Log the evaluation results for the dataset.

        :param metrics: The metrics for the dataset.
        The metrics are a dictionary with the metric name as key and the metric value as value.
        :param dataset_name: The name of the dataset.

        :return: None
        """

        logger.info(f"Results for: {dataset_name}")
        if self.writer is not None:
            self.writer.log(metrics, dataset_name)
        logger.info("-------------------------------------")
