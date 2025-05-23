from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Auction params
__C.q = 4
__C.constant = 3
__C.max_len = 150
__C.query = "有哪些适合夏天穿的清爽、时尚的女生连衣裙推荐？请给出不超过100字的回复"
__C.is_structured = True
__C.theta_1 = 0.5
__C.theta_2 = 0.6

# LLM model list
__C.models = ["Qwen"]
__C.auc = "Prom_ALS"
