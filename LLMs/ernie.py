import qianfan
from utils.api_config import require_env, request_timeout
from utils.base_llm import Base_LLM


class ernie(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super().__init__(config, data, auc_alg)
        require_env('QIANFAN_ACCESS_KEY', 'QIANFAN_SECRET_KEY')
        self.model, = require_env('ERNIE_MODEL')
        self.client = qianfan.ChatCompletion()
        self.timeout = request_timeout()

    def reply(self, query):
        self.messages.append({"role": "user", "content": query})
        response = self.client.do(model=self.model, messages=self.messages,
                                  request_timeout=self.timeout, retry_count=0)
        model_reply = response['body']['result']
        self.messages.append({"role": "assistant", "content": model_reply})
        return model_reply
