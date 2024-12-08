import os
import json
from model_zoo.model_zoo_ALS import ModelZoo_ALS
from cfgs.ADGEN_ALS_configs import config_jacket_ALS, config_skirt_ALS, config_trousers_ALS

config_map = {
    'jacket': config_jacket_ALS.cfg,
    'skirt': config_skirt_ALS.cfg,
    'trousers': config_trousers_ALS.cfg
}

name = ['jacket', 'skirt', 'trousers']
for i in range(1, 9):
    for j in range(3):
        file_path = f'dataset/ADGEN/{name[j]}/{name[j]}_section{i}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            cfg = config_map[name[j]]
            data_name = f"{name[j]}_section{i}"
            model_zoo = ModelZoo_ALS(cfg, data, data_name)
            model_zoo.run_all_models()
