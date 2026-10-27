# Evidence Set Decoding (ESD)

ESD selects an evidence set one sentence at a time. A shared cross-encoder scores the remaining sentences and a stopping action using the question and the evidence already selected. The same policy therefore decides both what to retain and when the context is complete.

## Code layout

| Path | Purpose |
| --- | --- |
| `src/esd.py` | Set-conditioned policy and training |
| `src/esd_run.py` | Greedy evidence-set decoding |
| `src/data_prep.py`, `src/units.py` | Dataset conversion and sentence units |
| `src/ftce_train.py` | Initial cross-encoder training |
| `src/reader.py`, `src/evaluate.py` | Reader inference and answer evaluation |
| `src/baselines.py`, `src/legacy/` | Comparison methods |
| `srv_src/`, `scripts/` | Additional experiment implementations and launch scripts |

Start with `src/` for the ESD training and decoding workflow.

## Prepare the environment and data

The core dependencies are CUDA-enabled PyTorch, Transformers, NumPy and pandas. Reader inference uses vLLM. Install the dependencies for the stages you intend to run in the same Python environment.

Run commands from the repository root. Place the source datasets under `data/raw/`, following the filenames and formats in `src/data_prep.py`. That script prepares HotpotQA, 2WikiMultihopQA and MuSiQue training examples, as well as evaluation splits that include Natural Questions:

```bash
python src/data_prep.py
```

Processed examples are stored in `data/proc/`. Each example contains a question and passages split into sentences, with evidence labels used during training.

## Train and decode

Download the initial reranker to `models/bge-reranker-v2-m3`, or change `--backbone` to your model directory. The following commands train the initial cross-encoder, fit ESD and decode the test sets:

```bash
python src/ftce_train.py --backbone models/bge-reranker-v2-m3 --out ckpt/ftce
python src/esd.py --backbone ckpt/ftce --train data/proc/train.jsonl --out ckpt/esd --seed 0
python src/esd_run.py --ckpt ckpt/esd --name esd --datasets hotpotqa,musique,2wiki
```

Decoding writes the decision sequences to `results/chains/<dataset>/` and the resulting reader contexts to `results/compressed/<dataset>/`. It also writes an equal-count marginal selection for comparing set-conditioned decoding with independent sentence scores.

Use `src/reader.py` to answer from the compressed contexts and `src/evaluate.py` to compute answer metrics. Reader model paths are supplied through the reader's command-line arguments. `--help` lists the options for each of the commands above after its Python dependencies are installed.

The repository includes source code and experiment scripts. Download datasets and pretrained models separately; trained checkpoints and generated results are not bundled.

## License

A license for the project-specific code has not yet been specified.
