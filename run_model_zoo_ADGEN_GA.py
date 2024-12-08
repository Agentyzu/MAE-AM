import os
import json
from model_zoo.model_zoo_GA import ModelZoo_GA
from cfgs.ADGEN_GA_configs import config_skirt_GA, config_jacket_GA, config_trousers_GA

config_map = {
    'jacket': config_jacket_GA.cfg,
    'skirt': config_skirt_GA.cfg,
    'trousers': config_trousers_GA.cfg
}

name = ['jacket', 'skirt', 'trousers']

for i in range(1, 10):
    for j in range(3):
        file_path = f'C:/Users/lenovo/Desktop/demo/dataset/ADGEN_25/{name[j]}/{name[j]}_section{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            cfg = config_map[name[j]]
            data_name = f"{name[j]}_section{i}"
            model_zoo = ModelZoo_GA(cfg, data, data_name)
            model_zoo.run_all_models()
