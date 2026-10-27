from utils.base_llm import Base_LLM
from openai import OpenAI
from utils.api_config import require_env, request_timeout


class being(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super(being, self).__init__(config, data, auc_alg)
        api_key, base_url, self.model = require_env('BEING_API_KEY', 'BEING_BASE_URL', 'BEING_MODEL')
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout(), max_retries=0)

    def reply(self, query):
        self.messages.append({
            "role": "user",
            "content": query
        })

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages
        )

        model_reply = completion.choices[0].message.content

        self.messages.append({
            "role": "assistant",
            "content": model_reply
        })

        return model_reply
