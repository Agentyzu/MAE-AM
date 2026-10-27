import json
from importlib import import_module
from pathlib import Path

from auction.Prom import Prom
from auction.Prom_DSIC import Prom_DSIC
from auction.GA import GA
from auction.GSP import GSP
from auction.GSP_SQA import GSP_SQA
from utils.LLMProcess import LLMProcess


class ModelZoo:
    available_llms = {name: name for name in
                      ("ernie", "kimi", "GLM", "Qwen", "ChatGPT", "claude", "being")}
    available_auc_alg = {
        "Prom": Prom, "GSP": GSP, "GA": GA, "GSP_SQA": GSP_SQA,
        "Prom_DSIC": Prom_DSIC, "Prom_AAL": Prom, "Prom_ALS": Prom
    }
    process_methods = {
        "Prom": "run", "GSP": "run", "GA": "run_ga", "GSP_SQA": "run_sqa",
        "Prom_DSIC": "run_DSIC", "Prom_AAL": "run_aal", "Prom_ALS": "run_als"
    }

    def __init__(self, cfg, data, data_name):
        self.cfg = cfg
        self.data = data
        self.data_name = data_name
        self.auc_alg = self.available_auc_alg[cfg["auc"]]
        self.llms = {}
        if not cfg.models:
            raise ValueError("Select at least one model adapter")
        for llm_name in cfg.models:
            self.add_model(llm_name)

    def run_all_models(self):
        highest_score = float('-inf')
        best_model = None
        for llm_name in self.llms:
            result = self.run_model(llm_name)
            if "score" not in result:
                continue
            print(f"The score of {llm_name}: {result['score']}")
            if result['score'] > highest_score:
                highest_score = result['score']
                best_model = llm_name
        if best_model is not None:
            print(f"{best_model} with the highest score {highest_score}")

    def package_results(self, result, llm_name):
        text = json.dumps(result, ensure_ascii=False, indent=4, allow_nan=False)
        root = Path(self.cfg.get("output_dir", Path(__file__).resolve().parents[1] / "results"))
        data_dir = root / self.cfg.get("dataset", "ADGEN") / self.cfg["auc"] / self.data_name
        data_dir.mkdir(parents=True, exist_ok=True)
        output_file = data_dir / f'{llm_name}_results_{self.data_name}.json'
        output_file.write_text(text, encoding='utf-8')
        print(f"Saved results: {output_file}")

    def run_model(self, llm_name):
        if llm_name not in self.llms:
            raise ValueError(f"Model adapter not enabled: {llm_name}")
        print(f"Run model {llm_name}:")
        llm_process = self.llms[llm_name]
        llm_process.result = {}
        llm_process.llm.messages = []
        method = self.process_methods[self.cfg["auc"]]
        getattr(llm_process, method)()
        llm_process.result["config"] = dict(self.cfg)
        llm_process.result["bids"] = llm_process.llm.b.tolist()
        self.package_results(llm_process.result, llm_name)
        return llm_process.result

    def add_model(self, llm_name):
        if llm_name not in self.available_llms:
            raise ValueError(f"Unknown model adapter: {llm_name}")
        if llm_name in self.llms:
            return
        module = import_module(f"LLMs.{llm_name}")
        model_class = getattr(module, self.available_llms[llm_name])
        self.llms[llm_name] = LLMProcess(model_class, self.cfg, self.data, self.auc_alg)

    def list_models(self):
        return list(self.llms)
