import os
import json
from model_zoo.model_zoo_ALS import ModelZoo_ALS
import importlib

source_dir = r"dataset\ATVI"
type_names = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]

config_map = {}
for name in type_names:
    config_module = importlib.import_module(f"cfgs.ATVI_ALS_configs.config_{name}_ALS")
    config_map[name] = config_module.cfg

for i in range(1, 10):
    for j in range(14):
        file_path = f'dataset/ATVI/{type_names[j]}/{type_names[j]}_section{i}.json'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            cfg = config_map[type_names[j]]
            data_name = f"{type_names[j]}_section{i}"
            model_zoo = ModelZoo_ALS(cfg, data, data_name)
            model_zoo.run_all_models()
