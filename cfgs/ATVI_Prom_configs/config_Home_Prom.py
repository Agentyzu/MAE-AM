from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Auction params
__C.q = 4
__C.constant = 3
__C.max_len = 150
__C.query = "What are some simple home improvement ideas for creating a more comfortable living space? Please keep the response under 50 words."
# __C.query = "What are some simple home improvement ideas for creating a more comfortable living space? Please keep the response under 50 words."
__C.is_structured = False
__C.theta_1 = 0.5
__C.theta_2 = 0.6

# LLM model list
__C.models = ["Qwen"]
# __C.models = ["ChatGPT", "Qwen", "GLM"]
__C.auc = "Prom"
