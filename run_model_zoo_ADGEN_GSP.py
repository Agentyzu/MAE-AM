import os
import json
from model_zoo.model_zoo import ModelZoo
from cfgs.ADGEN_GSP_configs import config_trousers_GSP, config_skirt_GSP, config_jacket_GSP

config_map = {
    'jacket': config_jacket_GSP.cfg,
    'skirt': config_skirt_GSP.cfg,
    'trousers': config_trousers_GSP.cfg
}

name = ['jacket', 'skirt', 'trousers']

for i in range(1, 10):
    for j in range(3):
        file_path = f'C:/Users/lenovo/Desktop/demo/dataset/ADGEN_50/{name[j]}/{name[j]}_section{i}.json'

        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            cfg = config_map[name[j]]
            data_name = f"{name[j]}_section{i}"
            model_zoo = ModelZoo(cfg, data, data_name)
            model_zoo.run_all_models()
