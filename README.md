<!-- markdownlint-disable MD033 -->
<h1 align="center">
  One Answer, Multiple Ads<br>
  Auction-Guided Advertising in LLM Responses
</h1>
<!-- markdownlint-enable MD033 -->

**MAE-AM** uses auctions to allocate advertising word-length budgets and embed multiple ads into a single LLM response. Rather than selecting only one advertisement, it distributes the available advertising space among competing advertisers and incorporates their content into a query-conditioned answer.

The framework connects auction-based allocation with response generation, enabling the study of advertiser utility and response quality across different auction mechanisms.

![Framework overview](fig.png)

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Configure

Set environment variables for the adapters you enable; no credentials or private endpoints are included.

| Adapter / service | Environment variables |
| --- | --- |
| OpenAI-compatible adapters | `<PREFIX>_API_KEY`, `<PREFIX>_BASE_URL`, `<PREFIX>_MODEL`; prefix: `QWEN`, `CHATGPT`, `CLAUDE`, `KIMI`, or `BEING` |
| GLM | `GLM_API_KEY`, `GLM_MODEL`; optional `GLM_BASE_URL` |
| ERNIE | `QIANFAN_ACCESS_KEY`, `QIANFAN_SECRET_KEY`, `ERNIE_MODEL` |
| Similarity | `SIMILARITY_API_KEY`, `SIMILARITY_SECRET_KEY`, `SIMILARITY_AUTH_URL`, `SIMILARITY_URL` |

`SIMILARITY_URL` is the complete similarity endpoint, without an access token. `API_TIMEOUT_SECONDS` defaults to 60. Keep credentials out of version control.

Edit queries and budgets in `cfgs/`. Data splits belong under `dataset/<DATASET>/<CATEGORY>/`. Language and the `content` / `ad_copy` field are selected automatically.

### 3. Run experiments

From the repository root:

```bash
python run_experiment.py --dataset ADGEN --auction Prom
python run_experiment.py --dataset ATVI --auction Prom
```

Auctions: `Prom`, `GSP`, `GA`, `SQA`, `ALS`, `AAL`, `Prom_DSIC`. By default, all available configured categories and splits are run. To reproduce a subset, specify it explicitly:

```bash
python run_experiment.py --dataset ADGEN --auction Prom --categories jacket --sections 1 --models Qwen --seed 0
```

Use `--data-dir` and `--output-dir` for custom directories; `--language` and `--content-field` override automatic selection. Existing module commands under `run_ADGEN/` and `run_ATVI/` accept the same options. Run `python run_experiment.py --help` for details.

### 4. Aggregate results

JSON results are saved to `results/<DATASET>/<AUCTION>/<SPLIT>/` with configuration and bids (`ALS` → `Prom_ALS`, `AAL` → `Prom_AAL`, `SQA` → `GSP_SQA`). Aggregation accepts the same auction names and exports per-split metrics, means, variances, and counts:

```bash
python -m utils.data_process --dataset ADGEN --auction Prom
python -B test_regressions.py
```

DSIC tables also report `advertiser_utility`, `truthful_utility` (the same advertiser at `beta=1`), and `utility_gain`; unselected advertisers have zero utility. A positive gain indicates a profitable deviation for that sample, not a proof of incentive compatibility.

Use `--results-dir` and `--output` to customize aggregation paths. Offline checks use Python 3.13.13; live provider calls and a clean installation of all SDKs have not been verified. A fixed seed controls local sampling, not provider-side generation.

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{anonymous2027oneanswer,
  title     = {One Answer, Multiple Ads: Auction-Guided Advertising in {LLM} Responses},
  author    = {Anonymous},
  booktitle = {Proceedings of the ACM Web Conference 2027 (WWW '27)},
  year      = {2027}
}
```
