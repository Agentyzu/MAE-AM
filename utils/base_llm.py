import numpy as np
from tqdm import tqdm
from utils.calculate_sim import sim


class Base_LLM(object):
    """
    A base class for implementing LLM-based advertisement integration.
    """
    def __init__(self, config, data, auc_alg):
        self.data = data
        self.query = config.query

        # Extracting advertisement data
        self.ad_name = [entry['ad_name'] for entry in data]
        self.pctr = [entry['pctr'] for entry in data]
        self.content = [entry['content'] for entry in data]    # ADGEN dataset
        # self.content = [entry['ad_copy'] for entry in data]  # ATVI dataset

        # Configuration parameters
        self.q = config.q
        self.constant = config.constant
        self.max_len = config.max_len
        self.is_structured = config.is_structured
        self.theta_1 = config.theta_1
        self.theta_2 = config.theta_2
        self.pos_norm = np.array([0.9 ** (k + 1) for k in range(self.q)])

        self.messages = []
        self.auc_alg = auc_alg
        self.basic_reply = None
        self.gen_reply = None
        self.sigma, self.prom, self.payments, self.utilities, self.SW = [None for _ in range(5)]
        self.user_satis, self.ad_satis, self.score = None, None, None
        self.b = None

    def calculate_similar(self, answer):
        """
        Calculates the similarity between a generated response and each advertisement's content.
        """
        n = len(self.content)
        similar = np.zeros(n)

        for i in tqdm(range(n)):
            words1 = answer.replace('\n', ' ').split()
            if len(words1) > 50:
                answer = ' '.join(words1[:50])

            words = self.content[i].replace('\n', ' ').split()
            if len(words) > 50:
                self.content[i] = ' '.join(words[:50])

            similar[i] = sim(answer, self.content[i])

        # Simulated similarity for testing
        self.b = np.random.rand(n)

    def generate_b(self):
        """
        Generates random bids for advertisements.
        """
        n = len(self.content)
        b_hat = np.random.uniform(0.2, 0.8, size=n)
        return b_hat

    def ad_gen_chinese(self):
        """
        Generates an advertisement-embedded response in Chinese, either structured or integrated.
        """
        if self.is_structured:
            prompt = (
                f"用户提问: \"{self.query}\"。请基于已经生成的基础回复来回答问题，并在基础回复中自然地融入广告推荐。\n"
                "每条广告内容必须严格按照分配的广告长度进行压缩，并且简明扼要，保留关键信息。\n"
                "输出格式为：\n"
                "关于用户提问的基础回复\n"
                "1. 广告商名称：根据指定长度压缩后的广告内容\n"
                "2. 广告商名称：根据指定长度压缩后的广告内容\n"
                "...\n"
                "广告内容必须按照以下长度限制进行压缩：\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 2:
                    prompt += (
                        f"第{k + 1}个广告位的广告商为：{self.ad_name[self.sigma[k]]}，"
                        f"原始广告内容为：{self.content[self.sigma[k]]}，"
                        f"请将其严格压缩为 {self.ad_len[k]} 字的内容。\n"
                    )

        else:
            prompt = (
                f"用户提问: \"{self.query}\"。请基于已经生成的基础回复来回答问题，并在基础回复中自然地融入广告推荐。\n"
                "广告内容必须经过压缩，保留关键信息，并确保广告的嵌入不会影响用户体验。\n"
                "广告应融入到回复中，而不是单独列出。\n"
                "广告内容必须按照以下长度限制进行压缩：\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 2:
                    prompt += (
                        f"第{k + 1}个广告位的广告商为：{self.ad_name[self.sigma[k]]}，"
                        f"原始广告内容为：{self.content[self.sigma[k]]}，"
                        f"请将其严格压缩为 {self.ad_len[k]} 字的内容。\n"
                    )

            prompt += (
                "请确保广告嵌入内容流畅，符合对话上下文，并且广告信息清晰可见。"
                "回复示例：\n"
                "[基础回复部分]，此外，我们还为您推荐了一些必备单品，"
                "如[广告商]的[根据指定长度压缩后的广告内容]，也可以考虑[广告商]的[根据指定长度压缩后的广告内容],...\n"
            )

        self.compressed_ad = self.reply(prompt)
        return self.compressed_ad

    def ad_gen_chinese_ALS(self):
        if self.is_structured:
            prompt = (
                "每条广告内容必须严格按照分配的广告长度进行压缩，并且简明扼要，保留关键信息。\n"
                "输出格式为：\n"
                "1. 广告商名称：根据指定长度压缩后的广告内容\n"
                "2. 广告商名称：根据指定长度压缩后的广告内容\n"
                "...\n"
                "广告内容必须按照以下长度限制进行压缩：\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 10:
                    prompt += (
                        f"第{k + 1}个广告位的广告商为：{self.ad_name[self.sigma[k]]}，"
                        f"原始广告内容为：{self.content[self.sigma[k]]}，"
                        f"请将其严格压缩为 {self.ad_len[k]} 字的内容。\n"
                    )

        else:
            prompt = (
                "广告内容必须经过压缩，保留关键信息，并确保广告的嵌入不会影响用户体验。\n"
                "广告应融入到回复中，而不是单独列出。\n"
                "广告内容必须按照以下长度限制进行压缩：\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 2:
                    prompt += (
                        f"第{k + 1}个广告位的广告商为：{self.ad_name[self.sigma[k]]}，"
                        f"原始广告内容为：{self.content[self.sigma[k]]}，"
                        f"请将其严格压缩为 {self.ad_len[k]} 字的内容。\n"
                    )

            prompt += (
                "请确保广告嵌入内容流畅，符合对话上下文，并且广告信息清晰可见。"
                "回复示例：[广告商]的[根据指定长度压缩后的广告内容]，也可以考虑[广告商]的[根据指定长度压缩后的广告内容],...\n"
            )

        self.compressed_ad = self.reply(prompt)
        return self.compressed_ad

    def ad_gen_english(self):
        """
        Generates a structured Chinese advertisement list with compressed content.
        """
        if self.is_structured:
            prompt = (
                f"User question: \"{self.query}\". Please respond based on the pre-generated basic response, naturally incorporating advertisement recommendations into it.\n"
                "Each advertisement content must be strictly compressed according to the allocated length, concise and to the point, preserving key information.\n"
                "The output format should be as follows:\n"
                "Basic response to the user question\n"
                "1. Advertiser name: Advertisement content compressed to the specified length\n"
                "2. Advertiser name: Advertisement content compressed to the specified length\n"
                "...\n"
                "The advertisements must be compressed according to the following length limits:\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 2:
                    prompt += (
                        f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                        f"original ad content: {self.content[self.sigma[k]]}, "
                        f"please strictly compress it to {self.ad_len[k]} characters.\n"
                    )

        else:
            prompt = (
                f"User question: \"{self.query}\". Please respond based on the pre-generated basic response, naturally incorporating advertisement recommendations.\n"
                "Advertisement content must be compressed, preserving key information and ensuring the integration does not disrupt the user experience.\n"
                "The ads should be embedded within the response, not listed separately.\n"
                "The advertisements must be compressed according to the following length limits:\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 2:
                    prompt += (
                        f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                        f"original ad content: {self.content[self.sigma[k]]}, "
                        f"please strictly compress it to {self.ad_len[k]} characters.\n"
                    )

            prompt += (
                "Ensure the ad integration is seamless, fits the conversational context, and that ad information is clearly visible.\n"
                "Response example:\n"
                "[Basic response part], additionally, we recommend some essential items, such as [Advertiser]'s [ad content compressed to the specified length], "
                "and also consider [Advertiser]'s [ad content compressed to the specified length],...\n"
            )

        self.compressed_ad = self.reply(prompt)
        return self.compressed_ad

    def ad_gen_english_ALS(self):
        if self.is_structured:
            prompt = (
                "Each advertisement content must be strictly compressed according to the allocated length, concise and to the point, preserving key information.\n"
                "The output format should be:\n"
                "1. Advertiser name: Advertisement content compressed to the specified length\n"
                "2. Advertiser name: Advertisement content compressed to the specified length\n"
                "...\n"
                "The advertisements must be compressed according to the following length limits:\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 10:
                    prompt += (
                        f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                        f"original ad content: {self.content[self.sigma[k]]}, "
                        f"please strictly compress it to {self.ad_len[k]} characters.\n"
                    )

        else:
            prompt = (
                "Advertisement content must be compressed, preserving key information, and ensuring the integration does not disrupt the user experience.\n"
                "The ads should be embedded within the response, not listed separately.\n"
                "The advertisements must be compressed according to the following length limits:\n"
            )

            for k in range(self.q):
                if self.ad_len[k] > 2:
                    prompt += (
                        f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                        f"original ad content: {self.content[self.sigma[k]]}, "
                        f"please strictly compress it to {self.ad_len[k]} characters.\n"
                    )

            prompt += (
                "Ensure the ad integration is seamless, fits the conversational context, and that ad information is clearly visible.\n"
                "Response example: [Advertiser]'s [ad content compressed to the specified length], and also consider [Advertiser]'s [ad content compressed to the specified length],...\n"
            )

        self.compressed_ad = self.reply(prompt)
        return self.compressed_ad

    def ad_gen_chinese_GA(self):
        reply = ""
        for i in range(np.sum(np.array(self.ad_len) > 10)):
            reply += f"{i + 1}.{self.content[self.sigma[i]][:self.ad_len[i]]};\n"
        self.compressed_ad = reply
        return self.compressed_ad

    def ad_gen_chinese_SQA(self):
        reply = self.basic_reply
        for i in range(np.sum(np.array(self.ad_len) > 10)):
            reply += f"{self.content[self.sigma[i]][:self.ad_len[i]]}"
        self.compressed_ad = reply
        return self.compressed_ad

    def ad_gen_english_GA(self):
        reply = ""
        for i in range(np.sum(np.array(self.ad_len) > 10)):
            words = self.content[self.sigma[i]].replace('\n', ' ').split()  # 按空格和换行符分割为单词
            truncated_content = ' '.join(words[:self.ad_len[i]])  # 按单词数限制
            reply += f"{i + 1}. {truncated_content};\n"
        self.compressed_ad = reply
        return self.compressed_ad

    def ad_gen_english_SQA(self):
        reply = self.basic_reply
        for i in range(np.sum(np.array(self.ad_len) > 10)):
            words = self.content[self.sigma[i]].replace('\n', ' ').split()  # 按空格和换行符分割为单词
            truncated_content = ' '.join(words[:self.ad_len[i]])  # 按单词数限制
            reply += f"{truncated_content} "
        self.compressed_ad = reply
        return self.compressed_ad

    def satisfaction(self):
        """
        Evaluates user satisfaction and advertisement satisfaction based on similarity and utility metrics.
        """
        def truncate_text(text, max_words):
            words = text.replace('\n', ' ').split()
            return ' '.join(words[:max_words]) if len(words) > max_words else ' '.join(words)

        ad_satis_k = np.zeros(self.q)
        for k in range(self.q):
            if self.ad_len[k] > 5:
                words1 = truncate_text(self.content[self.sigma[k]], max_words=50)
                words2 = truncate_text(self.compressed_ad, max_words=50)
                ad_satis_k[k] = sim(words1, words2)
            else:
                ad_satis_k[k] = 0

        self.user_satis = sim(truncate_text(self.basic_reply,50), words2)

        self.ad_satis = np.mean(ad_satis_k)
        self.score = self.theta_1 * self.user_satis + (1 - self.theta_1) * self.ad_satis + self.theta_2 * self.SW

    def reply(self, query):
        """
        Abstract method for generating replies to user queries.
        """
        raise NotImplementedError

    def auction(self):
        """
        Executes the auction algorithm to allocate ad slots based on given parameters.
        """
        len_content = [len(ads) for ads in self.content]
        self.sigma, self.prom, self.payments, self.utilities, self.SW = self.auc_alg(self.b, self.pctr, self.pos_norm,
                                                                                     self.q, self.constant, len_content,
                                                                                     self.max_len)
        self.ad_len = [int(self.max_len * p) for p in self.prom]

    def auction_DSIC(self, beta, s):
        """
        Executes a DSIC-compliant auction where one advertiser misreports their preferences.
        """
        len_content = [len(ads) for ads in self.content]
        self.sigma, self.prom, self.payments, self.utilities, self.SW = self.auc_alg(self.b, self.pctr, self.pos_norm,
                                                                                     self.q, self.constant, len_content,
                                                                                     self.max_len, beta, s)
        self.ad_len = [int(self.max_len * p) for p in self.prom]

