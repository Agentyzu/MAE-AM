from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Auction params
__C.q = 4
__C.constant = 3
__C.max_len = 150
__C.query = "What are effective ways to enhance skincare and beauty routines for different skin types? Please keep the response under 50 words."
# __C.query = "What are effective ways to enhance skincare and beauty routines for different skin types? Please keep the response under 50 words."
__C.is_structured = True
__C.theta_1 = 0.5
__C.theta_2 = 0.6

# LLM model list
__C.models = ["Qwen"]
# __C.models = ["ChatGPT", "claude", "ernie", "Qwen", "kimi", "GLM"]
__C.auc = "Prom_ALS"
