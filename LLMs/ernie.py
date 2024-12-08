import os
import qianfan
from utils.base_llm import Base_LLM


class ernie(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super(ernie, self).__init__(config, data, auc_alg)

    def reply(self, query):
        self.messages.append({
            "role": "user",
            "content": query
        })

        os.environ["QIANFAN_ACCESS_KEY"] = ""
        os.environ["QIANFAN_SECRET_KEY"] = ""
        chat_comp = qianfan.ChatCompletion()
        resp = chat_comp.do(model="ERNIE-4.0-8K", messages=self.messages)
        model_reply = resp["body"]['result']

        self.messages.append({
            "role": "assistant",
            "content": model_reply
        })

        return model_reply

