import os

from zhipuai import ZhipuAI
from utils.api_config import require_env, request_timeout
from utils.base_llm import Base_LLM


class GLM(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super().__init__(config, data, auc_alg)
        api_key, self.model = require_env('GLM_API_KEY', 'GLM_MODEL')
        options = {'api_key': api_key, 'timeout': request_timeout(), 'max_retries': 0}
        base_url = os.getenv('GLM_BASE_URL', '')
        if base_url:
            options['base_url'] = base_url
        self.client = ZhipuAI(**options)

    def reply(self, query):
        self.messages.append({"role": "user", "content": query})
        response = self.client.chat.completions.create(model=self.model, messages=self.messages)
        model_reply = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": model_reply})
        return model_reply
