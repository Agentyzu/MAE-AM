import json
import os

from LLMs.ernie import ernie
from LLMs.kimi import kimi
from LLMs.GLM import GLM
from LLMs.Qwen import Qwen
from LLMs.ChatGPT import ChatGPT
from LLMs.claude import claude
from LLMs.being import being

from auction.GSP_SQA import GSP_SQA
from model_zoo.model_zoo import ModelZoo
from utils.sqa_protocol import prepare

from utils.LLMProcess import LLMProcess


class ModelZoo_SQA(ModelZoo):
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
            "GSP_SQA": GSP_SQA
        }

        self.llms = {}
        self.auc_alg = self.available_auc_alg[self.cfg["auc"]]

        for llm_name in self.cfg.models:
            if llm_name in self.available_llms:
                self.llms[llm_name] = LLMProcess(self.available_llms[llm_name], self.cfg, self.data,
                                                 self.auc_alg)

    def _prepare_request(self, names=None):
        self.best_model = self.final_result = self.final_reply = None
        processes = self.llms if names is None else {name: self.llms[name] for name in names}
        if not processes:
            raise ValueError("Configure at least one supported SQA candidate model")
        self.base_model = next(iter(processes))
        self.shared_request, self.candidate_bank = prepare(processes)
        self.package_results({
            "record_type": "SQA_preauction_fusion_candidates",
            "economic_fields": "zero_placeholders_not_auction_results",
            "candidate_bank": {
                str(i): {name: cached["result"] for name, cached in models.items()}
                for i, models in self.candidate_bank.items()
            }
        }, "preauction")
        return self.shared_request

    def run_all_models(self):
        shared = self._prepare_request()
        winner = int(shared["sigma"][0])
        means = {}
        for name, process in self.llms.items():
            process.run_sqa(shared, self.candidate_bank[winner][name])
            process.result["base_model"] = self.base_model
            process.result["candidate_models"] = list(self.llms)
            self.package_results(process.result, name)
            means[name] = process.result["mean_score"]
        best = shared["sqa_setup"]["fixed_answers"][winner]["selected_model"]
        return self._select_final_reply(best, means)

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
        if llm_name not in self.llms:
            print(f"LLM {llm_name} not in model zoo")
            return None
        shared = self._prepare_request([llm_name])
        winner = int(shared["sigma"][0])
        process = self.llms[llm_name]
        process.run_sqa(shared, self.candidate_bank[winner][llm_name])
        process.result["base_model"] = self.base_model
        process.result["candidate_models"] = [llm_name]
        self.package_results(process.result, llm_name)
        return self._select_final_reply(llm_name, {llm_name: process.result["mean_score"]})

    def add_model(self, llm_name):
        if llm_name not in self.llms:
            self.llms[llm_name] = LLMProcess(self.available_llms[llm_name], self.cfg, self.data, self.auc_alg)
        else:
            print(f"Model {llm_name} exists.")

    def list_models(self):
        return list(self.llms.keys())
