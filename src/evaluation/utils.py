import logging
from heapq import nlargest
from typing import Dict, List, NamedTuple, Optional, Tuple

from src.dataset.beir.classes import CorpusDocument, RetrievalDataset
from src.rerankers.rank_llm.result import LLMRerankerResult
from src.utils import ResultDict

logger = logging.getLogger(__name__)


class Results(NamedTuple):
    """
    Class to store results in a structured way.
    """

    query_ids: List[str]
    queries: List[str]
    doc_ids: List[List[str]]
    passages: List[List[str]]


MetricDict = Dict[str, float]  # Dictionary to store metrics, e.g. {"recall@3": 0.5, "precision@3": 0.3}


def convert_dict_results_to_lists(results: ResultDict, dataset: RetrievalDataset) -> Results:
    """
    Converts a dict of dicts to a dict of lists.

    :param results: Dict of dicts as used in retriever and reranker
    :param dataset: Retrieval dataset
    :return: Tuple of query ids, queries, doc ids and passages
    """
    query_ids = list(results.keys())
    queries = [dataset.queries[query_id] for query_id in query_ids]
    doc_ids: List[List[str]] = [list(results[query_id].keys()) for query_id in query_ids]
    docs: List[List[CorpusDocument]] = [[dataset.corpus[doc_id] for doc_id in doc_ids[i]] for i in range(len(doc_ids))]
    passages: List[List[str]] = [["\n".join([doc["title"], doc["text"]]) for doc in doc_list] for doc_list in docs]

    assert len(query_ids) == len(queries) == len(doc_ids) == len(passages), (
        f"Lengths of query_ids, queries, doc_ids and passages do not match. "
        f"Got {len(query_ids)}, {len(queries)}, {len(doc_ids)} and {len(passages)} respectively."
    )

    for doc_list, passage_list in zip(doc_ids, passages):
        assert len(doc_list) == len(passage_list), (
            f"Lengths of doc_ids and passages do not match. "
            f"Got {len(doc_list)} and {len(passage_list)} respectively."
        )

    return Results(query_ids, queries, doc_ids, passages)


def convert_dict_results_to_llmrank_list_results(
    dict_results: ResultDict, dataset: RetrievalDataset
) -> List[LLMRerankerResult]:
    """
    Converts a dict retriever's results to a list of Results as used in the LLM reranker.

    :param dict_results: Dict of dicts as used in retriever and reranker
    :param dataset: Retrieval dataset
    :return: List of Results as used in the LLM reranker
    """
    rank_llm_results: List[LLMRerankerResult] = []
    for query_id, passage_id_score_dict in dict_results.items():
        query = dataset.queries[query_id]
        hits = []
        rank_index = 1
        sorted_passage_id_score = sorted(passage_id_score_dict.items(), key=lambda x: x[1], reverse=True)
        for passage_id, score in sorted_passage_id_score:
            doc = dataset.corpus[passage_id]
            # Probably only score and rank are used in the reranker, but we have access to all information, so pass it.
            # Looks like RankZephyr does not use title. Later maybe try and concatenate.
            hits.append(
                {
                    "content": doc["text"],
                    "qid": query_id,
                    "docid": passage_id,
                    "rank": rank_index,
                    "score": score,
                }
            )
            rank_index += 1

        rank_llm_results.append(LLMRerankerResult(query, hits))

    return rank_llm_results


def convert_llmrank_list_results_to_dict_results(
    llmrank_results: List[LLMRerankerResult],
) -> ResultDict:
    """
    Converts a list of Results as used in the LLM reranker to a dict of dicts as used in the retriever.

    :param llmrank_results: List of Results as used in the LLM reranker
    :return: Dict of dicts as used in the retriever and evaluation.
    """
    logger.debug(f"Converting LLM rank results to dict results. Example first result: {llmrank_results[0]}")
    dict_results: ResultDict = {}
    for result in llmrank_results:
        # Would be better to store query id in Result, but for now let's not change the Result class.
        query_id = result.hits[0]["qid"]  # All hits have the same query id.
        passage_id_score_dict = {}
        for hit in result.hits:
            passage_id_score_dict[hit["docid"]] = hit["score"]
        dict_results[query_id] = passage_id_score_dict

    logger.debug(
        f"Finished converting LLM rank results to dict results. "
        f"Example first result: {list(dict_results.keys())[0]}: {dict_results[list(dict_results.keys())[0]]}"
    )

    return dict_results


def split_dict_results(results: ResultDict, top_k: int) -> Tuple[ResultDict, ResultDict]:
    """
    Splits the results dict into 2 dicts. One with the top_k elements and one with the rest.

    :param results: Results in the same format as the output of the retriever
    :param top_k: Maximum number of elements to keep in the first dict.
    :return: Tuple of dicts with the top_k elements and the rest.
    """
    assert top_k is not None and top_k > 0, f"Top_k must be a positive integer, got {top_k}."

    top_k_results: ResultDict = {query_id: {} for query_id in results.keys()}
    rest_results: ResultDict = {query_id: {} for query_id in results.keys()}

    def _safe_nlargest(top_k: int, scores: Dict[str, float]) -> Optional[float]:
        if not scores.values():
            return None
        top_k = min(top_k, len(scores.values()))
        return nlargest(top_k, scores.values())[-1]

    k_scores: Dict[str, float] = {query_id: _safe_nlargest(top_k, scores) for query_id, scores in results.items()}

    for query_id in results.keys():
        for doc_id, score in results[query_id].items():
            # Make sure we don't overflow the top_k results in top_k_results
            if score >= k_scores[query_id] and len(top_k_results[query_id]) < top_k:
                top_k_results[query_id][doc_id] = score
            else:
                rest_results[query_id][doc_id] = score

    return top_k_results, rest_results


def merge_dict_results(results1: ResultDict, results2: ResultDict, override: bool = False) -> ResultDict:
    """
    Merges 2 dicts of dicts.
    Important when the results of the reranker need to be merged with the results of the retriever.

    :param results1: First dict of dicts
    :param results2: Second dict of dicts
    :param override: If True, the scores from the second dict will override the scores from the first dict.
    :return: Merged dict of dicts
    """
    merged_results = results1.copy()
    for query_id, query_results in results2.items():
        assert query_id in merged_results, f"Query id {query_id} not in results1"

        for doc_id, score in query_results.items():
            if override:
                merged_results[query_id][doc_id] = score
            else:
                if doc_id not in merged_results[query_id]:
                    merged_results[query_id][doc_id] = score

    return merged_results


def scale_dict_results(results1: ResultDict, results2: ResultDict, eps: float = 0.01) -> ResultDict:
    """
    Scales the scores of the second dict of dicts to the range of the first dict of dicts.
    The highest score for passages in the results2 will be lower than the lowest score for passages in respective query
    in results1.

    :param results1: First dict of dicts. Used to scale the second dict of dicts.
    :param results2: Second dict of dicts. Will be scaled to the range of the first dict of dicts.
    :param eps: Epsilon value to subtract from the highest score in results2 when scaling so that the max in results2
    is strictly lower than the min in results1 for respective query.
    :return: Scaled results2
    """
    results1_mins_dict: Dict[str, float] = {
        query_id: min(scores.values(), default=0) for query_id, scores in results1.items()
    }
    results2_maxs_dict: Dict[str, float] = {
        query_id: max(scores.values(), default=0) for query_id, scores in results2.items()
    }

    scaled_results2 = results2.copy()
    for query_id in results2.keys():
        for doc_id, score in results2[query_id].items():
            scaled_results2[query_id][doc_id] = (
                score - results2_maxs_dict[query_id] + results1_mins_dict[query_id] - eps
            )

    return scaled_results2


def inform_about_copy_error(copy_exit_code: int) -> None:
    """
    Function checks whether after copying models, the exit code was 0 or not.
    Function only checks and logs, does not throw exception, as probably without working copying
    the code will still work as the model will be downloaded again.

    :param copy_exit_code: Exit code of the copy command.
    """
    if copy_exit_code != 0:
        logger.error(f"Copying the model from /nas failed with exit code {copy_exit_code}.")


def merge_list_of_retrieval_results(list_of_results: List[ResultDict], override: bool = False) -> ResultDict:
    """
    Merges the results of multiple retrievers into one dict of dicts. The merged results are appropriately scaled.
    :param list_of_results: List of results to merge. Starts with the results of the best retriever and ends with the
    results of the worst retriever.
    :param override: If True, the scores from the second dict will override the scores from the first dict in case of conflicts of scores.
    :return: Merged dict of dicts
    """
    # Not elegant, but potentially in some places with reranker this function might be called with the dict
    # which should not be merged.
    # TODO investigate later, as this check should be redundant (not it is necessary).
    if not isinstance(list_of_results, list):
        logger.warning("Not merging as the input is not a list. Returning the input.")
        return list_of_results

    # Additional checks
    if not list_of_results:
        logger.warning("Empty list provided. Returning an empty dictionary.")
        return {}

    if not all(isinstance(result, dict) for result in list_of_results):
        logger.warning("Not merging as the input is not a list of dictionaries. Returning the input.")
        return list_of_results

    # Merge results
    original_retriever_results = list_of_results[0]
    for i in range(1, len(list_of_results)):
        logger.info("Merging results of the ensemble retriever.")
        # Scale the results - we assume that retrievers in the ensemble are in order of best to worst.
        scaled_results = scale_dict_results(original_retriever_results, list_of_results[i])
        # In case of a tie, the original retriever should not be overridden not to change the score.
        original_retriever_results = merge_dict_results(original_retriever_results, scaled_results, override=override)

    return original_retriever_results
