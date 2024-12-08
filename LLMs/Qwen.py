import json
import os
from openai import OpenAI
from base_llm import Base_LLM


class Qwen(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super(Qwen, self).__init__(config, data, auc_alg)

    def reply(self, query):
        self.messages.append({
            "role": "user",
            "content": query
        })

        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=self.messages,
        )

        data = json.loads(completion.model_dump_json())
        model_reply = data['choices'][0]['message']['content']

        self.messages.append({
            "role": "assistant",
            "content": model_reply
        })

        return model_reply
