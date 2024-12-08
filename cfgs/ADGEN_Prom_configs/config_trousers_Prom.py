from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Auction params
__C.q = 4
__C.constant = 3
__C.max_len = 150
# __C.query = "秋季男士牛仔裤在舒适性和耐用性方面如何体现优势？有哪些材质和设计能够满足不同场合的需求？请给出不超过100字的回复"
__C.query = "有哪些适合秋天穿的舒适且耐用的男生牛仔裤推荐？请给出不超过100字的回复"
__C.is_structured = True
__C.theta_1 = 0.5
__C.theta_2 = 0.6

# LLM model list
__C.models = ["Qwen"]
# __C.models = ["ChatGPT", "Qwen", "kimi", "GLM"]
__C.auc = "Prom"
