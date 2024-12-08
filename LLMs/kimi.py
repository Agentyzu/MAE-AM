from openai import OpenAI
from utils.base_llm import Base_LLM


class kimi(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super(kimi, self).__init__(config, data, auc_alg)

    def reply(self, query):
        apikey = ""
        client = OpenAI(
            api_key=apikey,
            base_url="https://api.moonshot.cn/v1",
        )

        self.messages.append({
            "role": "user",
            "content": query
        })

        completion = client.chat.completions.create(
            model="moonshot-v1-32k",
            messages=self.messages,
            temperature=0.3,
        )

        model_reply = completion.choices[0].message.content

        self.messages.append({
            "role": "assistant",
            "content": model_reply
        })

        return model_reply
