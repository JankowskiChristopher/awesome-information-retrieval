from src.dataset.qa.dataset_configs import MATHQA, MEDMCQA, OPENBOOKQA, TRUTHFULQA, WIKIQA
from src.evaluation.hf_eval_modules.metric_defs import compute_exact_match, compute_f1

NAS_DIR = "/nas/llm_domain_adaptation"
LOCAL_DIR = "./storage"

MODELS_ROOT_DIR = "models"
TOKENIZERS_ROOT_DIR = "tokenizers"
DATASETS_ROOT_DIR = "dataset"
QA_DATASETS_ROOT_DIR = "dataset/qa"
CHECKPOINT_ROOT_DIR = "checkpoint"

RESULTS_SUBFOLDER = "scored_docs"
BEIR_DATASETS = [
    "trec-covid",
    "nfcorpus",
    "scifact",
    "scidocs",
    "arguana",
    "fiqa",
    "webis-touche2020",
    "dbpedia-entity",
    "quora",
    "nq",
    "hotpotqa",
    "cqadupstack",
    "climate-fever",
    "msmarco",
    "fever",
    "germanquad",
]

CQADUPSTACK_SUBFORUMS = [
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
]


CQADUPSTACK_SUBFORUMS = [
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
]


QA_DATASETS = {
    "truthfulqa": TRUTHFULQA,
    "openbookqa": OPENBOOKQA,
    "wikiqa": WIKIQA,
    "medmcqa": MEDMCQA,
    "mathqa": MATHQA,
}
QA_METRICS = {"em": compute_exact_match, "f1": compute_f1}
