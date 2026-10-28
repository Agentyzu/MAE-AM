import json
import os
from copy import deepcopy

from LLMs.ernie import ernie
from LLMs.kimi import kimi
from LLMs.GLM import GLM
from LLMs.Qwen import Qwen
from LLMs.ChatGPT import ChatGPT
from LLMs.claude import claude
from LLMs.being import being

from auction.Prom import Prom
from auction.Prom_DSIC import Prom_DSIC
from auction.GA import GA
from auction.PA_AIGC import PA_AIGC
from auction.LERA import LERA
from auction.GSP import GSP
from auction.GSP_SQA import GSP_SQA

from utils.LLMProcess import LLMProcess


class ModelZoo:
    def __new__(cls, cfg, data, data_name):
        if cls is ModelZoo:
            if cfg["auc"] == "GSP_SQA":
                from model_zoo.model_zoo_SQA import ModelZoo_SQA
                return object.__new__(ModelZoo_SQA)
            if cfg["auc"] == "Prom_DSIC":
                from model_zoo.model_zoo_DSIC import ModelZoo_DSIC
                return ModelZoo_DSIC(cfg, data, data_name)
        return super().__new__(cls)

    def __init__(self, cfg, data, data_name):
        self.cfg = cfg
        self.data = data
        self.data_name = data_name

        self.available_llms = {
            "ernie": ernie,
            "kimi": kimi,
            "GLM": GLM,
            "Qwen": Qwen,
            "ChatGPT": ChatGPT,
            "claude": claude,
            "being": being
        }

        self.available_auc_alg = {
            "Prom": Prom,
            "GSP": GSP,
            "GA": GA,
            "GSP_SQA": GSP_SQA,
            "Prom_DSIC": Prom_DSIC,
            "PA_AIGC": PA_AIGC,
            "LERA": LERA
        }

        self.llms = {}
        self.auc_alg = self.available_auc_alg[self.cfg["auc"]]

        for llm_name in self.cfg.models:
            if llm_name in self.available_llms:
                self.llms[llm_name] = LLMProcess(self.available_llms[llm_name], self.cfg, self.data,
                                                 self.auc_alg)

    def bind_instance(self, instance):
        from utils.experiment_inputs import bind
        return bind(self, instance)

    def _prepare_request(self):
        self.best_model = None
        self.final_result = None
        self.final_reply = None
        if not self.llms:
            raise ValueError("Configure at least one supported candidate model")
        self.base_model = next(iter(self.llms))
        self.shared_request = self.llms[self.base_model].prepare_request()
        return self.shared_request

    def run_all_models(self):
        highest_score = float('-inf')
        best_model = None
        model_mean_scores = {}
        shared_request = self._prepare_request()

        for llm_name, llm_process in self.llms.items():
            print(f"Run model {llm_name}:")
            llm_process.run(shared_request)
            llm_process.result["base_model"] = self.base_model
            llm_process.result["candidate_models"] = list(self.llms)
            self.package_results(llm_process.result, llm_name)

            mean_score = llm_process.result["mean_score"]
            model_mean_scores[llm_name] = mean_score
            print(f"The mean score of {llm_name} : {mean_score}")

            if mean_score > highest_score:
                highest_score = mean_score
                best_model = llm_name

        print(f"{best_model} with the highest mean score {highest_score}")
        return self._select_final_reply(best_model, model_mean_scores)

    def _select_final_reply(self, llm_name, model_mean_scores):
        winner = self.llms[llm_name].result
        best = winner["samples"][winner["best_sample_index"]]
        self.final_result = {
            key: deepcopy(winner[key]) for key in (
                "query", "basic_reply", "ad_bids", "ad_pctr", "ad_rank",
                "ad_prom", "ad_sw", "ad_rev", "ad_len", "ad_utility",
                "base_model", "candidate_models", "n_samples", "dataset", "language",
                "length_unit", "length_counting_rule"
            )
        }
        for key in ("auction", "retrieval", "sqa_setup", "experiment", "basic_reply_provenance", "generation_config"):
            if key in winner:
                self.final_result[key] = deepcopy(winner[key])
        self.final_result.update(deepcopy(best))
        self.final_result.update({
            "selected_model": llm_name,
            "selected_sample_index": best["sample_index"],
            "selected_model_mean_score": winner["mean_score"],
            "model_mean_scores": dict(model_mean_scores),
            "score_aggregation": "selected_sample"
        })
        self.best_model = llm_name
        self.final_reply = best["gen_reply"]
        self.package_results(self.final_result, "selected")
        return self.final_reply

    def package_results(self, result, llm_name):
        result_dir = f'result_{self.cfg["auc"]}'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        data_dir = os.path.join(result_dir, self.data_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        output_file = os.path.join(data_dir, f'{llm_name}_results_{self.data_name}.json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

    def run_model(self, llm_name):
        if llm_name in self.llms:
            print(f"Run model {llm_name}:")
            shared_request = self._prepare_request()
            self.llms[llm_name].run(shared_request)
            self.llms[llm_name].result["base_model"] = self.base_model
            self.llms[llm_name].result["candidate_models"] = [llm_name]
            self.package_results(self.llms[llm_name].result, llm_name)
            return self._select_final_reply(
                llm_name, {llm_name: self.llms[llm_name].result["mean_score"]}
            )
        else:
            print(f"LLM {llm_name} not in model zoo")

    def add_model(self, llm_name):
        if llm_name not in self.llms:
            self.llms[llm_name] = LLMProcess(self.available_llms[llm_name], self.cfg, self.data, self.auc_alg)
        else:
            print(f"Model {llm_name} exists.")

    def list_models(self):
        return list(self.llms.keys())
