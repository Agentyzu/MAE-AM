import json
import os

from LLMs.ernie import ernie
from LLMs.kimi import kimi
from LLMs.GLM import GLM
from LLMs.Qwen import Qwen
from LLMs.ChatGPT import ChatGPT
from LLMs.claude import claude
from LLMs.being import being

from functools import partial
from auction.ALS import ALS
from model_zoo.model_zoo import ModelZoo

from utils.LLMProcess import LLMProcess


class ModelZoo_ALS(ModelZoo):
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
            "Prom_ALS": partial(ALS, beta=self.cfg.get("als_beta", 0.5))
        }

        self.llms = {}
        self.auc_alg = self.available_auc_alg[self.cfg["auc"]]

        for llm_name in self.cfg.models:
            if llm_name in self.available_llms:
                self.llms[llm_name] = LLMProcess(self.available_llms[llm_name], self.cfg, self.data,
                                                 self.auc_alg)

    def run_all_models(self):
        highest_score = float('-inf')
        best_model = None

        shared_request = self._prepare_request()
        model_mean_scores = {}
        for llm_name, llm_process in self.llms.items():
            print(f"Run model {llm_name}:")
            llm_process.run_als(shared_request)
            llm_process.result["base_model"] = self.base_model
            llm_process.result["candidate_models"] = list(self.llms)
            model_mean_scores[llm_name] = llm_process.result["mean_score"]
            self.package_results(llm_process.result, llm_name)

            print(f"The score of {llm_name} : {llm_process.result['score']}")

            if llm_process.result['score'] > highest_score:
                highest_score = llm_process.result['score']
                best_model = llm_name

        print(f"{best_model} with the highest mean score {highest_score}")
        return self._select_final_reply(best_model, model_mean_scores)

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
            process = self.llms[llm_name]
            process.run_als(shared_request)
            process.result["base_model"] = self.base_model
            process.result["candidate_models"] = [llm_name]
            self.package_results(process.result, llm_name)
            return self._select_final_reply(llm_name, {llm_name: process.result["mean_score"]})
        else:
            print(f"LLM {llm_name} not in model zoo")

    def add_model(self, llm_name):
        if llm_name not in self.llms:
            self.llms[llm_name] = LLMProcess(self.available_llms[llm_name], self.cfg, self.data, self.auc_alg)
        else:
            print(f"Model {llm_name} exists.")

    def list_models(self):
        return list(self.llms.keys())
