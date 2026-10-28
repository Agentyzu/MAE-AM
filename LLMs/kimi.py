from utils.base_llm import Base_LLM
from utils.backbones import reply


class kimi(Base_LLM):
    backend_name = "kimi"

    def __init__(self, config, data, auc_alg):
        super(kimi, self).__init__(config, data, auc_alg)

    def reply(self, query):
        return reply(self, query, "kimi")
