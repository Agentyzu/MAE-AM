from utils.base_llm import Base_LLM
from utils.backbones import reply


class ChatGPT(Base_LLM):
    backend_name = "ChatGPT"

    def __init__(self, config, data, auc_alg):
        super(ChatGPT, self).__init__(config, data, auc_alg)

    def reply(self, query):
        return reply(self, query, "ChatGPT")
