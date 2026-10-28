from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.q = 4
__C.constant = 3
__C.max_len = 150
__C.query = "What strategies can help raise awareness about responsible alcohol consumption? Please keep the response under 50 words."
__C.is_structured = True
__C.theta_1 = 0.5
__C.theta_2 = 0.6

__C.models = ["Qwen"]
__C.auc = "GSP"
