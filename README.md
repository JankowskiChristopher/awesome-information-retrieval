# Awesome Information Retrieval
<p align="left">
<a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Created by: [Krzysztof Jankowski](https://github.com/jankowskichristopher), [Michał Janik](https://github.com/mihal09), [Michał Grotkowski](https://github.com/mgrotkowski), [Antoni Hanke](https://github.com/AntekHanke).
The research was presented at the Data Science Summit ML Edition 2024 and earlier parts at ML in PL 2024 and Warsaw.ai episode XX.

### About
The repository contains code developed for experimentation with information retrieval and question answering systems.
By combining various retrievers, rerankers and other techniques we conduct an in depth analysis on how to achieve the most performant pipelines.
The repository uses the following models:
Retrievers:
- [BM25](https://pypi.org/project/rank-bm25/) and ElasticSearch BM25
- [Dragon](https://huggingface.co/facebook/dragon-plus-context-encoder)
- [Snowflake Arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)

Rerankers:
- [BGE-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- [Rank Zephyr](https://huggingface.co/castorini/rank_zephyr_7b_v1_full)
- Own reranker - Mistral 7B with special prompt to compare question with 2 passages
- Hybrid rerankers: flexible code to combine rerankers into pipelines or split the retrievers results into several rerankers

Other models can be easily integrated through Hydra configs.


Technical paper coming soon.


### Running
The experiments were run on a Kubernetes cluster. We provide the instructions in:
- [Kubernetes Job Deployement](/docs/jobs.md)
Configuration files are available in [Kubernetes jobs](/jobs/).
In order to use them some fields e.g. VolumeMount need to be modified as they are specific to the cluster.

### Standard Python Installation and Execution

To set up and run the project locally, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/jankowskichristopher/awesome-information-retrieval.git
   cd awesome-information-retrieval
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Change to the source directory and run the main script:
   ```
   cd src
   python main.py
   ```

Make sure you have Python 3.x installed on your system before following these steps.

### Project structure
The repository uses Hydra to efficiently manage different [configs](/src/cfgs/) and override parameters.
The conducted experiments with results are [reported](/experiments/) alongside [plots](/experiments/plots) and useful scripts for visualization and data processing. More information about visualization is present in a separate [README](/experiments/README.md).

The source code is divided into a structure that enables relatively easy modifications.

```
.
├── src/
│   ├── cfgs/                 # Hydra configs
│   ├── dataset/
│   │   ├── beir/             # BEIR dataset for retrieval evaluation
│   │   └── qa/               # Different question answering datasets for evaluation of generators
│   ├── evaluation/           # For the retrieval and generation evaluation
│   ├── experiments/          # Code for conducting various experiments
│   ├── generators/           # Various LLM generators used in generation and LLM reranking
│   ├── rerankers/            # Various rerankers e.g. embedding and LLM
│   └── retrievers/           # Various retrievers e.g. BM25, Dragon, Arctic
├── experiments/
│   ├── plots/                # Plots for visualization
│   └── README.md             # More information about visualization
├── constants.py              # Useful constants
├── utils.py                  # Useful utils
├── constants.py              # Weights and Biases logging
└── README.md

```
