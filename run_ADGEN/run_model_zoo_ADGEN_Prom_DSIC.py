import os
import json
from model_zoo.model_zoo_DSIC import ModelZoo_DSIC
from cfgs.ADGEN_Prom_configs import config_jacket_Prom, config_skirt_Prom, config_trousers_Prom

config_map = {
    'jacket': config_jacket_Prom.cfg,
    'skirt': config_skirt_Prom.cfg,
    'trousers': config_trousers_Prom.cfg
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
            model_zoo = ModelZoo_DSIC(cfg, data, data_name)
            model_zoo.run_all_models()
