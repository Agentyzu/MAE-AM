from utils.base_llm import Base_LLM
from utils.backbones import reply


class GLM(Base_LLM):
    backend_name = "GLM"

    def __init__(self, config, data, auc_alg):
        super(GLM, self).__init__(config, data, auc_alg)

    def reply(self, query):
        return reply(self, query, "GLM")
