from zhipuai import ZhipuAI
from base_llm import Base_LLM


class GLM(Base_LLM):
    def __init__(self, config, data, auc_alg):
        super(GLM, self).__init__(config, data, auc_alg)

    def reply(self, query):
        self.messages.append({
            "role": "user",
            "content": query
        })

        client = ZhipuAI(api_key="")  # 请填写您自己的APIKey
        response = client.chat.completions.create(
            model="glm-4",  # 请填写您要调用的模型名称
            messages=self.messages,
        )

        model_reply = response.choices[0].message.content

        self.messages.append({
            "role": "assistant",
            "content": model_reply
        })

        return model_reply
