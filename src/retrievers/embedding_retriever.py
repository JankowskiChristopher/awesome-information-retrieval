import gc
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import faiss
import numpy as np
import torch
from datasets import Dataset
from llama_index.embeddings import BaseEmbedding, HuggingFaceEmbedding
from llama_index.embeddings.huggingface_utils import format_query
from tqdm import tqdm
from transformers import AutoTokenizer

from src.constants import DATASETS_ROOT_DIR
from src.dataset.beir.classes import CorpusType
from src.retrievers.base_retriever import BaseRetriever
from src.utils import batch_iterable, escape_name, get_absolute_path, get_model, get_tokenizer

logger = logging.getLogger(__name__)


def index_path_factory(datasets_absolute_dir, dataset_name, index_filename):
    return Path(datasets_absolute_dir) / dataset_name / "indexes" / index_filename


class EmbeddingRetriever(BaseRetriever):
    def __init__(
        self,
        model_identifier: str,
        tokenizer_path_or_name: str,
        context_embedding_model_path_or_name: str,
        query_embedding_model_path_or_name: Optional[str] = None,
        batch_size_passages: int = 20,
        batch_size_queries: int = 256,
        batch_size_retrieval: int = 1000,
        embedding_column: str = "embeddings",
        metric_type=faiss.METRIC_INNER_PRODUCT,
        should_normalize_embeddings: bool = False,
        query_instruction: Optional[str] = None,
        context_instruction: Optional[str] = None,
        device: Optional[str] = None,
        local: bool = False,
        should_cache_results: bool = True,
        should_use_cached_results: bool = True,
        should_override_cache_with_less_docs: bool = False,
    ):
        super().__init__(should_cache_results, should_use_cached_results, should_override_cache_with_less_docs)
        logging.info("Initalizing Embedding retriever")

        self.corpus: Optional[CorpusType] = None
        self.dataset_name: Optional[str] = None
        self.dataset: Optional[Dataset] = None
        self.index_name: Optional[str] = None
        self.query_embedding_map: Optional[Dict[str, np.ndarray]] = None

        self.tokenizer: Optional[AutoTokenizer] = None
        self.context_embedding_model: Optional[BaseEmbedding] = None
        self.query_embedding_model: Optional[BaseEmbedding] = None

        self.context_embedding_model_path_or_name = context_embedding_model_path_or_name
        self.query_embedding_model_path_or_name = (
            query_embedding_model_path_or_name or context_embedding_model_path_or_name
        )
        self.tokenizer_path_or_name = tokenizer_path_or_name
        self.model_identifier = model_identifier

        self.batch_size_retrieval = batch_size_retrieval
        self.batch_size_passages = batch_size_passages
        self.batch_size_queries = batch_size_queries
        self._embedding_column = embedding_column

        self.query_instruction = query_instruction
        self.context_instruction = context_instruction

        self.device = device
        self.local = local
        self.metric_type = metric_type
        self.should_normalize_embeddings = should_normalize_embeddings

    def get_name(self) -> str:
        return f"EmbeddingRetriever_{self.model_identifier}"

    def index_corpus(self, corpus: CorpusType, dataset_name: str, dataset: Optional[Dataset] = None):
        self.corpus = corpus
        self.dataset_name = dataset_name
        self.dataset = dataset

        self.ensure_dataset_initialized()
        self.initialize_index_name()
        self.load_or_create_index()

    def ensure_dataset_initialized(self):
        if self.dataset is None:
            logger.info("Creating huggingface dataset")

            self.dataset = Dataset.from_dict(
                {
                    "doc_id": self.corpus.keys(),
                    "corpus": [
                        self.format_document(doc_id, doc["title"], doc["text"], self.dataset_name)
                        for doc_id, doc in self.corpus.items()
                    ],
                    "title": [doc["title"] for doc in self.corpus.values()],
                }
            )

    def initialize_index_name(self):
        model_name = escape_name(self.model_identifier)
        logger.info(f"Initializing FAISS index for dataset {self.dataset_name} and model {model_name}")
        self.index_name = f"{model_name}_index"

    def load_or_create_index(self):
        index_path = self.get_index_path()

        if self.index_exists(index_path):
            self.load_index(index_path)
        else:
            self.compute_embeddings_if_needed()
            self.create_and_save_index(index_path)

    def get_index_path(self):
        datasets_absolute_dir = get_absolute_path(DATASETS_ROOT_DIR, local=self.local)
        return index_path_factory(datasets_absolute_dir, self.dataset_name, f"{self.index_name}.faiss")

    def index_exists(self, index_path):
        return index_path.exists()

    def load_index(self, index_path):
        logger.info(f"FAISS index exists, loading from {index_path}")
        self.dataset.load_faiss_index(self.index_name, index_path)

    def compute_embeddings_if_needed(self):
        if self._embedding_column not in self.dataset.column_names:
            logger.info(f"Calculating embeddings, batch_size={self.batch_size_passages}")
            start_time = time.perf_counter()

            self.load_context_embedding_model()

            self.dataset = self.dataset.map(
                lambda x: {self._embedding_column: np.array(self.get_corpus_embeddings(x["corpus"]))},
                batched=True,
                batch_size=self.batch_size_passages,
            )

            logger.info(f"Finished embedding passages. Elapsed time: {time.perf_counter() - start_time} seconds")
            self.free_context_embedding_model()

    def create_and_save_index(self, index_path: str):
        logger.info(f"Creating FAISS index and saving it to {index_path}")
        self.dataset.add_faiss_index(
            column=self._embedding_column, index_name=self.index_name, metric_type=self.metric_type
        )
        self.dataset.save_faiss_index(self.index_name, index_path)

    def process_queries(self, query_ids: Iterable[str], queries: Iterable[str], batch_size: Optional[int] = None):
        if self.index_name is None:
            raise RuntimeError("Index is not initialized. Please run index_corpus before using this method.")

        if batch_size is None:
            batch_size = self.batch_size_queries

        query_embedding_map = {}

        batched_query_ids = batch_iterable(query_ids, batch_size)
        batched_queries = batch_iterable(queries, batch_size)

        logger.info(f"Starting embedding queries with batch_size={batch_size}")
        self.load_query_embedding_model()
        for queries_ids_batch, queries_batch in tqdm(zip(batched_query_ids, batched_queries)):
            queries_embeddings = np.array(self.get_query_embeddings(queries_batch), dtype=np.float32)
            for query_id, query_embedding in zip(queries_ids_batch, queries_embeddings):
                query_embedding_map[query_id] = query_embedding

        self.free_query_embedding_model()
        self.query_embedding_map = query_embedding_map

    def search(self, query, top_k) -> Dict[str, float]:
        if self.index_name is None:
            raise RuntimeError("Index is not initialized. Please run index_corpus before using this method.")

        if not self.query_embedding_model:
            logger.warning("Generating embeddings in search, please invoke process_queries first!")
            self.load_query_embedding_model()

        query_embedding = np.array(self.get_query_embeddings([query]), dtype=np.float32)

        closest_passages = self.dataset.get_nearest_examples(self.index_name, query_embedding, top_k)

        results = {
            doc_id: float(score) for doc_id, score in zip(closest_passages.examples["doc_id"], closest_passages.scores)
        }

        return results

    def search_batch(self, query_ids: List[str], queries: List[str], top_k) -> List[Dict[str, float]]:
        if self.index_name is None:
            raise RuntimeError("Index is not initialized. Please run index_corpus before using this method.")

        if not self.query_embedding_map:
            if not self.query_embedding_model:
                logger.warning("Generating embeddings in search_batch, please invoke process_queries first!")
                self.load_query_embedding_model()
            query_embeddings = np.array(self.get_query_embeddings(queries), dtype=np.float32)
        else:
            query_embeddings = np.stack([self.query_embedding_map[query_id] for query_id in query_ids])

        closest_passages_batch = self.dataset.get_nearest_examples_batch(self.index_name, query_embeddings, top_k)

        results = [
            {doc_id: float(score) for doc_id, score in zip(examples["doc_id"], scores)}
            for examples, scores in zip(closest_passages_batch.total_examples, closest_passages_batch.total_scores)
        ]

        return results

    def get_query_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Get query embeddings."""

        queries = [self.format_query(query) for query in queries]
        return self.query_embedding_model._embed(queries)

    def get_corpus_embeddings(self, documents: List[str]) -> List[List[float]]:
        """
        Calculate embeddings for a list of documents from the corpus.

        :param documents: List of documents (strings) to embed
        :return: List of embeddings
        """
        # Here we don't format the document as we are already formatting it in the ensure_dataset_initialized method
        return self.context_embedding_model._embed(documents)

    def format_document(self, doc_id: str, title: str, text: str, dataset_name: Optional[str] = None) -> str:
        """
        Format a document into a string for the embedding model.

        :param doc_id: Document ID
        :param title: Document title
        :param text: Document content
        :param dataset_name: Optional dataset name. Default is None.

        :return: Formatted document string for the embedding model
        """
        formatted_document = f"title: {title}\ndoc_id: {doc_id}\n\n{text}"
        if self.context_instruction:
            formatted_document = f"{self.context_instruction} {formatted_document}"
        return formatted_document

    def format_query(self, query: str) -> str:
        """
        Format a query string using an inferred instruction.
        Firstly, the instruction is loaded from Llamaindex if 'model_name' has an associated instruction.
        Otherwise, it directly uses 'self.query_instruction', if it is available;
        Finally, it loads the instruction from 'self.query_embedding_model' if it is available.

        :param query: A string representing the query to be formatted.
        :return: A string that is the formatted query.
        """

        # Attempt to retrieve the model name; defaults to None if 'query_embedding_model' is not initialized or doesn't have 'model_name'
        model_name = getattr(self.query_embedding_model, "model_name", None)

        # Use 'self.query_instruction' if available; otherwise, attempt to use instruction from 'query_embedding_model'
        query_instruction = (
            self.query_instruction
            if self.query_instruction
            else getattr(self.query_embedding_model, "instruction", None)
        )
        return format_query(query, model_name, query_instruction)

    def load_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = get_tokenizer(self.tokenizer_path_or_name, local=self.local)

    def free_tokenizer(self):
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_context_embedding_model(self):
        if self.context_embedding_model is None:
            self.load_tokenizer()
            huggingface_context_embed_model = get_model(
                self.context_embedding_model_path_or_name, device=self.device, local=self.local
            )
            context_embed_model = HuggingFaceEmbedding(
                model=huggingface_context_embed_model,
                tokenizer=self.tokenizer,
                normalize=self.should_normalize_embeddings,
                model_name=self.model_identifier,
            )
            context_embed_model.normalize = self.should_normalize_embeddings
            context_embed_model.embed_batch_size = self.batch_size_passages

            self.context_embedding_model = context_embed_model

    def free_context_embedding_model(self):
        if self.context_embedding_model:
            del self.context_embedding_model
            self.context_embedding_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_query_embedding_model(self):
        if self.query_embedding_model is None:
            self.load_tokenizer()
            huggingface_query_embed_model = get_model(
                self.query_embedding_model_path_or_name, device=self.device, local=self.local
            )
            query_embed_model = HuggingFaceEmbedding(
                model=huggingface_query_embed_model,
                tokenizer=self.tokenizer,
                normalize=self.should_normalize_embeddings,
            )
            query_embed_model.normalize = self.should_normalize_embeddings
            self.query_embedding_model = query_embed_model

    def free_query_embedding_model(self):
        if self.query_embedding_model:
            del self.query_embedding_model
            self.query_embedding_model = None
            gc.collect()
            torch.cuda.empty_cache()

    def __exit__(self, *args, **kwargs):
        logger.info("__exit__ EmbeddingRetriever. Memory cleanup.")
        self.free_tokenizer()
        self.free_context_embedding_model()
        self.free_query_embedding_model()
