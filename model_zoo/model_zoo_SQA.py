from model_zoo.model_zoo import ModelZoo


class ModelZoo_SQA(ModelZoo):
    process_methods = {"GSP_SQA": "run_sqa"}
