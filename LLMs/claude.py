from utils.base_llm import Base_LLM
from utils.backbones import reply


class claude(Base_LLM):
    backend_name = "claude"

    def __init__(self, config, data, auc_alg):
        super(claude, self).__init__(config, data, auc_alg)

    def reply(self, query):
        return reply(self, query, "claude")
