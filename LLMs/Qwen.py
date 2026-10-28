from utils.base_llm import Base_LLM
from utils.backbones import reply


class Qwen(Base_LLM):
    backend_name = "Qwen"

    def __init__(self, config, data, auc_alg):
        super(Qwen, self).__init__(config, data, auc_alg)

    def reply(self, query):
        return reply(self, query, "Qwen")
