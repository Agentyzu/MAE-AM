from base_llm import Base_LLM
from openai import OpenAI


class claude(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super(claude, self).__init__(config, data, auc_alg)

    def reply(self, query):
        self.messages.append({
            "role": "user",
            "content": query
        })

        api_key = ""
        api_base = "https://sg.uiuiapi.com/v1"
        client = OpenAI(api_key=api_key, base_url=api_base)

        completion = client.chat.completions.create(
            model="claude-3-haiku-20240307",
            messages=self.messages
        )

        model_reply = completion.choices[0].message.content

        self.messages.append({
            "role": "assistant",
            "content": model_reply
        })

        return model_reply