import os
import json
from model_zoo.model_zoo_SQA import ModelZoo_SQA
from cfgs.ADGEN_SQA_configs import config_jacket_SQA, config_skirt_SQA, config_trousers_SQA

config_map = {
    'jacket': config_jacket_SQA.cfg,
    'skirt': config_skirt_SQA.cfg,
    'trousers': config_trousers_SQA.cfg
}

name = ['jacket', 'skirt', 'trousers']

for i in range(1, 10):
    for j in range(3):
        file_path = f'dataset/ADGEN/{name[j]}/{name[j]}_section{i}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            cfg = config_map[name[j]]
            data_name = f"{name[j]}_section{i}"
            model_zoo = ModelZoo_SQA(cfg, data, data_name)
            model_zoo.run_all_models()
