import json
import logging
import os
import re
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.constants import DATASETS_ROOT_DIR, LOCAL_DIR, MODELS_ROOT_DIR, NAS_DIR, RESULTS_SUBFOLDER, TOKENIZERS_ROOT_DIR

ResultDict = Dict[str, Dict[str, float]]

logger = logging.getLogger(__name__)


def get_absolute_path(path: str, local: bool = False):
    """
    Returns the absolute path of a file or directory. The main directory is either or /nas or a local one.
    :param path: Relative path
    :param local: If True, returns a local path, else returns a path in /nas
    :return: Absolute path
    """
    main_dir = LOCAL_DIR if local else NAS_DIR
    absolute_path = os.path.join(main_dir, path)
    return absolute_path


def get_model(
    model_name: str,
    device: str = "cuda",
    local: bool = False,
    auto_model_classification: bool = False,
    question_answering: bool = False,
) -> AutoModel:
    """
    Loads a model from the local directory if exists, otherwise downloads and saves it.
    Depending on the task either an AutoModel or AutoModelForSequenceClassification is loaded.
    The model is put on the device.
    Due to a bizarre bug when loading the model for sequence classification from disk - last layers are randomly
    initialized - the model is always downloaded from HuggingFace which fixes the bug.
    :param model_name: Name of the model
    :param device: Device to put the model on. Default is cuda.
    :param local: If True, loads from local directory, else loads from /nas
    :param auto_model_classification: If True, loads an AutoModelForSequenceClassification, else loads an AutoModel
    :param question_answering: If True, loads an AutoModelForCausalLM, else loads an AutoModel
    :return: Model
    """

    assert not (
        auto_model_classification and question_answering
    ), "Only one of auto_model_classification and question_answering can be True"

    logger.info(f"Loading model {model_name}")
    models_absolute_dir = get_absolute_path(MODELS_ROOT_DIR, local=local)
    model_path = Path(models_absolute_dir) / model_name

    # Handle AutoModelForSequenceClassification separately and always download from HuggingFace.
    if auto_model_classification:
        logger.info(f"Loading model for sequence classification")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        if model_path.exists():
            logger.info(f"Model exists, loading from {model_path}")
            if question_answering:
                config = AutoConfig.from_pretrained(model_path)
                logger.info(f"Loading a model for question answering in {config.torch_dtype}")
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=config.torch_dtype)
            else:
                model = AutoModel.from_pretrained(model_path)
        else:
            # Download and save the model
            logger.info("Model does not exist, downloading from Huggingface")

            if question_answering:
                config = AutoConfig.from_pretrained(model_name)
                logger.info(f"Model for question answering detected, downloading in {config.torch_dtype}")
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=config.torch_dtype)
            else:
                model = AutoModel.from_pretrained(model_name)

            # Save the model to the local directory
            model.save_pretrained(model_path)

    return model.to(device)


def get_tokenizer(tokenizer_name: str, local: bool = False) -> AutoTokenizer:
    """
    Loads a tokenizer from the local directory if exists, otherwise downloads and saves it.
    :param tokenizer_name: Name of the tokenizer
    :param local: If True, loads from local directory, else loads from /nas
    :return: Tokenizer
    """
    # Load the model from the stored files if exists, otherwise download and save
    logger.info(f"Loading tokenizer {tokenizer_name}")

    tokenizers_absolute_dir = get_absolute_path(TOKENIZERS_ROOT_DIR, local=local)
    tokenizer_path = Path(tokenizers_absolute_dir) / tokenizer_name

    if tokenizer_path.exists():
        logger.info(f"Tokenizer exists, loading from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Download and save the tokenizer
        logger.info("Tokenizer does not exist, downloading from Huggingface")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(tokenizer_path)

    return tokenizer


def escape_name(name: str):
    """
    Escapes a name by replacing '/' and '\' with '_'.
    Used to escape model names when saving to disk in order to be saved correctly.
    :param name: Name of the model
    :return: Name with escaped characters
    """
    return name.replace("/", "_").replace("\\", "_")


def flatten_list(lst: List[List[Any]]):
    """
    Flattens a list of lists to a single list.
    :param lst: List of lists to be flattened.
    :return: Flattened list
    """
    return [item for sublist in lst for item in sublist]


def batch_list(lst, batch_size: int):
    """
    Converts a list to a list of lists with batch_size elements.
    Works on multidimensional lists,
    :param lst: Multidimensional list to be batched
    :param batch_size: Integer batch size
    :return: List with one more dimension which is the batch.
    The last batch may be smaller than batch_size because of truncation.
    """
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def batch_iterable(iterable, batch_size: int):
    """
    Converts an iterable to an iterator of lists with batch_size elements.
    Works on any iterable,
    :param iterable: Iterable to be batched
    :param batch_size: Integer batch size
    :return: An iterator with lists of up to batch_size elements each.
    The last batch may be smaller than batch_size because of truncation.
    """
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def get_result_path(
    model_name: str,
    dataset_name: str,
    subfolder: str = RESULTS_SUBFOLDER,
    num_passages: Optional[int] = None,
    local: bool = False,
) -> Path:
    """
    Construct the file path for storing or loading retrieval results based on the given parameters.

    This function generates a Path object representing the location where model results are to be saved or
    have been saved. It constructs the path using a root directory, dataset name, subfolder, and model name.
    It can adjust the path to be either in a local directory or a remote location based on the 'local' flag.

    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :param subfolder: Subfolder where the results are stored
    :param local: If True, saves to local directory, else saves to /nas
    :return: Path object representing the location where results are stored or loaded
    """
    datasets_absolute_dir = get_absolute_path(DATASETS_ROOT_DIR, local=local)

    file_name = f"{model_name}.json" if num_passages is None else f"{model_name}_{num_passages}.json"
    path = Path(datasets_absolute_dir) / dataset_name / subfolder / file_name
    return path


def save_dict_results(
    dict_results: ResultDict,
    model_name: str,
    dataset_name: str,
    subfolder: str = RESULTS_SUBFOLDER,
    num_passages: Optional[int] = None,
    should_override_cache_with_less_docs: bool = False,
    local: bool = False,
) -> None:
    """
    Save the dictionary containing retrieval results to a JSON file.

    This function serializes the results dictionary into a JSON file at a specified location,
    creating necessary directories if they do not exist. The path to save the results is constructed
    based on the model and dataset names, along with a specified subfolder.

    :param dict_results: Dictionary containing retrieval results
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :param subfolder: Subfolder where the results are stored
    :param num_passages: Number of passages contained in the results. Used to construct the file name.
    :param local: If True, saves to local directory, else saves to /nas
    :return: None
    """
    results_path = get_result_path(
        model_name=model_name, dataset_name=dataset_name, subfolder=subfolder, num_passages=num_passages, local=local
    )

    results_path.parent.mkdir(exist_ok=True, parents=True)

    if not should_override_cache_with_less_docs and results_path.exists():
        with open(results_path, "r") as file:
            old_results = json.load(file)

        # get the number of passages in the new results
        num_passages_new = max(len(documents) for documents in dict_results.values())
        num_passages_old = max(len(documents) for documents in old_results.values())

        # Check if the number of passages in the new results is less than the old results
        if num_passages_new <= num_passages_old:
            logger.info(
                f"Number of passages in the new results is less than the old results. "
                f"Old results will not be overridden. "
                f"New: {num_passages_new}, Old: {num_passages_old}"
            )
            return
        else:
            logger.info(
                f"Number of passages in the new results is greater or equal than the old results. "
                f"Overriding old results. "
                f"New: {num_passages_new}, Old: {num_passages_old}"
            )

    logger.info(f"Overriding cached results at {results_path}.")
    with open(results_path, "w") as file:
        json.dump(dict_results, file)

    logger.info(f"Successfully stored results for model {model_name} at {results_path}!")


def load_dict_results(
    model_name: str,
    dataset_name: str,
    top_k: Optional[int] = None,
    subfolder: str = RESULTS_SUBFOLDER,
    local: bool = False,
) -> Optional[ResultDict]:
    """
    Load results from a specified path, returning top k results if specified.
    If top_k is specified, only the top k results are returned. If the number of passages is less than k, a warning is logged and None is returned.
    If the file does not exist, a warning is logged and None is returned.

    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :param top_k: Number of top results to return. If None, all results are returned. Default is None.
    :param subfolder: Subfolder where the results are stored
    :param local: If True, loads from local directory, else loads from /nas
    :return: Dictionary of results. If the file does not exist or top k is specified and the number of loaded passages is less than k, returns None.
    """
    legacy_results_path = get_result_path(
        model_name=model_name, dataset_name=dataset_name, subfolder=subfolder, local=local
    )  # Legacy results path without number_of_passages in the filename
    filename_without_extension, file_extension = legacy_results_path.stem, legacy_results_path.suffix

    matching_files = []
    # Check if there are files with the model name and the number_of_passages in the filename
    if legacy_results_path.parent.exists():
        for file_name in os.listdir(legacy_results_path.parent):
            match = re.match(rf"{filename_without_extension}_(\d+)\{file_extension}", file_name)
            if match:
                matching_files.append((file_name, int(match.group(1))))

    # Sort matching files based on number_of_passages in descending order
    matching_files.sort(key=lambda x: x[1], reverse=True)

    highest_passages_num = None
    if matching_files:
        _, highest_passages_num = matching_files[0]  # Load the file with the highest number_of_passages
        results_path = get_result_path(
            model_name=model_name,
            dataset_name=dataset_name,
            subfolder=subfolder,
            num_passages=highest_passages_num,
            local=local,
        )
    else:
        #  if no files with pattern are found, fallback to loading legacy filename (without the number of passages)
        results_path = legacy_results_path

    if not results_path.exists():
        logger.warning(f"Failed to load results from {results_path}!")
        return None

    with open(results_path, "r") as file:
        dict_results = json.load(file)

    is_legacy = results_path == legacy_results_path
    if top_k:
        try:
            if is_legacy:
                # If there are no files with the new naming pattern, save the results with the new naming pattern and check the number of passages in get_top_k_results
                num_passages = min(len(documents) for documents in dict_results.values())
                save_dict_results(
                    dict_results, model_name, dataset_name, subfolder, num_passages=num_passages, local=local
                )

                # For legacy results, the number of passages is not stored in the file name, so we have to check it here
                dict_results = get_top_k_results(dict_results, top_k, check_passage_count=True)
            else:
                # If the file is not legacy (number of passages is present in the filename), we can ignore checking the number of passages in get_top_k_results
                if highest_passages_num < top_k:
                    return None
                dict_results = get_top_k_results(dict_results, top_k, check_passage_count=False)
        except ValueError as e:
            logger.warning(e)
            return None

    logger.info(f"Successfully loaded results for model {model_name} from {results_path}!")

    return dict_results


def get_top_k_results(dict_results: ResultDict, k: int, check_passage_count=True) -> ResultDict:
    """
    Returns the top k results from dict_results based on document scores.
    If k is less than or equal to 0, the function raises a ValueError.
    If check_passage_count is True and the number of passages in dict_results for any document is less than k, the function raises a ValueError.

    :param dict_results: Dictionary of results
    :param k: Number of top results to return
    :param check_passage_count: If True, checks the number of passages in dict_results. Default is True.

    """
    if k <= 0:
        raise ValueError("k must be greater than 0!")

    if check_passage_count:
        assert_dict_num_passages(dict_results, k, lambda passage_count, top_k: top_k <= passage_count)

    top_k_results = {}
    for query, documents in dict_results.items():
        sorted_documents = dict(sorted(documents.items(), key=lambda item: item[1], reverse=True)[:k])
        top_k_results[query] = sorted_documents

    return top_k_results


def assert_len_list_equal(list1: List[Any], list2: List[Any], message: Optional[str] = None) -> None:
    """
    Function to assert that the length of two lists are equal.
    :param list1: First list to compare
    :param list2: Second list to compare
    :param message: Optional message to display if the assertion fails. Otherwise, a default message is displayed.
    """

    if message is None:
        message = f"Length of lists do not match. " f"Got {len(list1)} and {len(list2)} respectively."
    assert len(list1) == len(list2), message


def assert_dict_results_equal(dict1: Dict[str, Dict[str, Any]], dict2: Dict[str, Dict[str, Any]]) -> None:
    """
    Function to assert that two dictionaries in the format used for BEIR evaluation are equal.
    First query keys are compare and later the passage ids are compared.
    All passage ids will be checked and the function will inform how many were mismatched.
    :param dict1: First dictionary to compare
    :param dict2: Second dictionary to compare
    """
    logger.info("Asserting that the dictionaries are equal.")
    dict1_keys = set(dict1.keys())
    dict2_keys = set(dict2.keys())
    assert dict1_keys == dict2_keys, (
        "Reranker should not change the keys of the results." f"but got query keys {dict1_keys} and {dict2_keys}"
    )

    wrong_keys = []
    for retriever_key in dict1_keys:
        retriever_passages_keys = set(dict1[retriever_key].keys())
        reranker_passages_keys = set(dict2[retriever_key].keys())
        if retriever_passages_keys != reranker_passages_keys:
            wrong_keys.append(retriever_key)

    assert not wrong_keys, (
        f"Passage ids do not match for {len(wrong_keys)} keys. First key {wrong_keys[0]}.\n"
        f"dict1 passages keys: {set(dict1[wrong_keys[0]].keys())}\n"
        f"dict2 passages keys: {set(dict2[wrong_keys[0]].keys())}\n"
    )


def assert_dict_contained(superset: Dict[str, Dict[str, Any]], subset: Dict[str, Dict[str, Any]]) -> None:
    """
    Function to assert that subset dict which has a form used in BEIR evaluation contains keys
    of the superset dict. Used when reranker reduces the number of retrieved passages.
    :param superset: Superset dictionary to compare. Usually results of the retriever
    :param subset: Subset dictionary to compare. Usually results of the reranker
    """
    logger.info("Asserting that the subset is contained in the superset.")
    superset_keys = set(superset.keys())
    subset_keys = set(subset.keys())
    assert superset_keys == subset_keys, (
        "Reranker should not change the keys of the results." f"but got query keys {superset_keys} and {subset_keys}"
    )

    wrong_keys = []
    for retriever_key in superset_keys:
        retriever_passages_keys = set(superset[retriever_key].keys())
        reranker_passages_keys = set(subset[retriever_key].keys())
        if not reranker_passages_keys.issubset(retriever_passages_keys):
            wrong_keys.append(retriever_key)

    assert not wrong_keys, (
        f"Passage ids do not match for {len(wrong_keys)} keys. First key {wrong_keys[0]}.\n"
        f"dict1 passages keys: {set(superset[wrong_keys[0]].keys())}\n"
        f"dict2 passages keys: {set(subset[wrong_keys[0]].keys())}\n"
    )


def assert_dict_num_passages(
    result_dict: Dict[str, Dict[str, Any]],
    num_passages: int,
    condition: Callable[[int, int], bool] = (lambda passage_count, num_passages: passage_count == num_passages),
) -> None:
    """
    Function to assert that the number of passages in the result_dict meets a certain condition based on the num_passages.
    If the condition is not met, a ValueError is raised.

    :param result_dict: Dictionary of results
    :param num_passages: Number of passages
    :param condition: comparison function used to compare the number of passages in the result_dict with num_passages. Default is equality.
    """
    for _, documents in result_dict.items():
        passage_count = len(documents)
        if not condition(passage_count, num_passages):
            raise ValueError(
                f"Condition failed for number of passages. "
                f"passage_count: {passage_count}, num_passages: {num_passages}"
            )


def check_total_gpu_memory(mode_function: Callable[[List[float]], float] = min) -> float:
    """
    Function to check total GPU memory available on the system.
    Handles multiple GPUs and uses mode_function to calculate the memory in the way specified by the user
    :param mode_function: Function to calculate the memory. Default is min.
    Typical uses cases might be min, max, mean, first.
    :return: GPU memory result from the mode_function . Result is in GiB.
    """
    import torch

    if torch.cuda.is_available():
        devices_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
        return mode_function(devices_memory) / (1024**3)
    else:
        logger.warning("CUDA is not available. Please make sure that you have installed PyTorch with CUDA support.")
        return 0.0
