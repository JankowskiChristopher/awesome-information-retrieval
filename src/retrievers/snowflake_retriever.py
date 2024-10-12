import logging
from typing import Optional

from src.retrievers.embedding_retriever import EmbeddingRetriever

logger = logging.getLogger(__name__)


class ArcticEmbedRetriever(EmbeddingRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.info("Initalizing ArcticEmbed retriever")

        if not self.query_instruction:
            self.query_instruction = "Represent this sentence for searching relevant passages:"

    def format_document(self, doc_id, title, text, dataset_name: Optional[str] = None) -> str:
        if self.dataset_name in ["quora"]:
            # For quora dataset, we need to format the corpus the same way as the queries
            # https://github.com/Snowflake-Labs/arctic-embed/commit/3c9d754a4497cea1e5e56361cdb0d588143c66f8#diff-d135986ed841e0e9a3d5bfd751d01b5cc97cd3155eb4cab5a0a81c95efd0c342R63
            return self.format_query(text)
        else:
            return f"{title} {text}"
