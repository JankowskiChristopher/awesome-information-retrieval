# Information Retrieval Playground

<p align="left">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Created by: Krzysztof Jankowski, Michał Janik, Michał Grotkowski, Antoni Hanke
### About
The repository contains code developed for experimentation with information retrieval and question answering systems.
By combining various retrievers, rerankers and other techniques we conduct an in depth analysis on how to achieve the most performant pipelines.
The repository uses the following models:
Retrievers:
- BM25
- Dragon
- Snowflake Arctic-embed

Rerankers:
- BGE-reranker
- Rank Zephyr
- Own reranker - Mistral 7B with special prompt to compare question with 2 passages

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
   cd information-retrieval-playground
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
