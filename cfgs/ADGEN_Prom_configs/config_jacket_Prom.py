from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Auction params
__C.q = 4
__C.constant = 3
__C.max_len = 150
__C.query = "秋天有哪些适合男生的时尚且舒适的牛仔外套推荐？请给出不超过100字的回复"
# __C.query = "秋季的牛仔外套对于男生的时尚搭配有哪些独特之处？如何在舒适与时尚之间找到平衡？请给出不超过100字的回复"
__C.is_structured = True
__C.theta_1 = 0.5
__C.theta_2 = 0.6

# LLM model list
__C.models = ["Qwen"]
# __C.models = ["ChatGPT", "Qwen", "kimi", "GLM"]
__C.auc = "Prom"
