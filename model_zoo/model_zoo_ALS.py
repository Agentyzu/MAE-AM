from model_zoo.model_zoo import ModelZoo


class ModelZoo_ALS(ModelZoo):
    process_methods = {"Prom_ALS": "run_als"}
