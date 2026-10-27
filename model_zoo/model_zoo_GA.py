from model_zoo.model_zoo import ModelZoo


class ModelZoo_GA(ModelZoo):
    process_methods = {"GA": "run_ga"}
