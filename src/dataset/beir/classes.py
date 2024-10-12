from dataclasses import dataclass
from typing import Dict, TypedDict


class CorpusDocument(TypedDict):
    title: str
    text: str


class QrelDocument(TypedDict):
    document_id: str
    relevance: int


CorpusType = Dict[str, CorpusDocument]  # Dict[document_id, document]
QueriesType = Dict[str, str]  # Dict[query_id, query_text]
QrelsType = Dict[str, QrelDocument]


@dataclass
class RetrievalDataset:
    corpus: CorpusType
    queries: QueriesType
    qrels: QrelsType
