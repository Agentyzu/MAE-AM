import numpy as np
from utils.calculate_sim import get_scorer
from auction.GA import GA
from utils.ad_spans import extract_ad_spans, span_instructions
from utils.text_length import dataset_settings, count_text, prefix_text, RULES


class Base_LLM(object):
    def __init__(self, config, data, auc_alg):
        self.data = data
        self.backbones = config.get("backbones", {})
        self.llm_calls = []
        self.basic_reply_provenance = []
        self.query = config.query

        self.ad_name = [entry['ad_name'] for entry in data]
        self.pctr = [entry['pctr'] for entry in data]
        self.content = [entry['content'] if 'content' in entry else entry['ad_copy']
                        for entry in data]
        self.dataset, self.language, self.length_unit = dataset_settings(config, data)
        self.length_counting_rule = RULES[self.length_unit]

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
        self.scorer = get_scorer()
        self.raw_gen_reply = None
        self.ad_spans = []
        self.span_diagnostics = []
        self._span_reply = None

    def calculate_similar(self, answer):
        self.b = np.random.rand(len(self.content))

    def _record_reply(self, raw_reply):
        self.raw_gen_reply = raw_reply
        indices = self.sigma[:self.q]
        if self.auc_alg is GA:
            indices = indices[self.prom[:self.q] > 0]
        self.compressed_ad, self.ad_spans, self.span_diagnostics = extract_ad_spans(
            raw_reply, indices
        )
        self._span_reply = self.compressed_ad
        for k, record in enumerate(self.ad_spans):
            generated = (count_text(record["text"], self.length_unit)
                         if record["status"] in ("ok", "empty") else None)
            target = int(self.ad_len[k])
            record.update(target_length=target, generated_length=generated,
                          length_error=generated - target if generated is not None else None,
                          length_unit=self.length_unit)
        if (self.messages and self.messages[-1].get("role") == "assistant"
                and self.messages[-1].get("content") == raw_reply):
            self.messages[-1]["content"] = self.compressed_ad
        return self.compressed_ad

    def _generate_with_spans(self, prompt, language):
        if language == "en":
            prompt += ("\nWrite the final reply in English. Count words as nonempty whitespace-separated segments, "
                       "including advertiser names but excluding attribution tags. For a zero target, emit an empty ad annotation.\n")
        else:
            prompt += ("\n请用中文回答。字数按Unicode字符计，包含广告商名称、标点、数字、字母和内部空白，"
                       "不计归属标注。目标为0时输出空的广告标注。\n")
        prompt += span_instructions(self.sigma[:self.q], self.ad_name, language)
        return self._record_reply(self.reply(prompt))

    def organic_prompt(self):
        instruction = "Please answer in English." if self.language == "en" else "请用中文回答。"
        return f"{self.query}\n{instruction}"

    def generate_ads(self, style="normal"):
        if self.q == 0:
            return self._record_reply(self.basic_reply)
        if style == "dwls":
            from utils.als_generation import generate_dwls
            return generate_dwls(self)
        suffix = {"normal": "", "als": "_ALS", "ga": "_GA", "sqa": "_SQA"}[style]
        language = "english" if self.language == "en" else "chinese"
        return getattr(self, f"ad_gen_{language}{suffix}")()

    def generate_b(self):
        n = len(self.content)
        b_hat = np.random.uniform(0.2, 0.8, size=n)
        return b_hat

    def ad_gen_chinese(self):
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

        return self._generate_with_spans(prompt, "zh")

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
                prompt += (
                    f"第{k + 1}个广告位的广告商为：{self.ad_name[self.sigma[k]]}，"
                    f"原始广告内容为：{self.content[self.sigma[k]]}，"
                    f"请将其严格压缩为 {self.ad_len[k]} 字的内容。\n"
                )

            prompt += (
                "请确保广告嵌入内容流畅，符合对话上下文，并且广告信息清晰可见。"
                "回复示例：[广告商]的[根据指定长度压缩后的广告内容]，也可以考虑[广告商]的[根据指定长度压缩后的广告内容],...\n"
            )

        return self._generate_with_spans(prompt, "zh")

    def ad_gen_english(self):
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
                prompt += (
                    f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                    f"original ad content: {self.content[self.sigma[k]]}, "
                    f"please strictly compress it to {self.ad_len[k]} words.\n"
                )

        else:
            prompt = (
                f"User question: \"{self.query}\". Please respond based on the pre-generated basic response, naturally incorporating advertisement recommendations.\n"
                "Advertisement content must be compressed, preserving key information and ensuring the integration does not disrupt the user experience.\n"
                "The ads should be embedded within the response, not listed separately.\n"
                "The advertisements must be compressed according to the following length limits:\n"
            )

            for k in range(self.q):
                prompt += (
                    f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                    f"original ad content: {self.content[self.sigma[k]]}, "
                    f"please strictly compress it to {self.ad_len[k]} words.\n"
                )

            prompt += (
                "Ensure the ad integration is seamless, fits the conversational context, and that ad information is clearly visible.\n"
                "Response example:\n"
                "[Basic response part], additionally, we recommend some essential items, such as [Advertiser]'s [ad content compressed to the specified length], "
                "and also consider [Advertiser]'s [ad content compressed to the specified length],...\n"
            )

        return self._generate_with_spans(prompt, "en")

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
                prompt += (
                    f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                    f"original ad content: {self.content[self.sigma[k]]}, "
                    f"please strictly compress it to {self.ad_len[k]} words.\n"
                )

        else:
            prompt = (
                "Advertisement content must be compressed, preserving key information, and ensuring the integration does not disrupt the user experience.\n"
                "The ads should be embedded within the response, not listed separately.\n"
                "The advertisements must be compressed according to the following length limits:\n"
            )

            for k in range(self.q):
                prompt += (
                    f"For ad slot {k + 1}, the advertiser is: {self.ad_name[self.sigma[k]]}, "
                    f"original ad content: {self.content[self.sigma[k]]}, "
                    f"please strictly compress it to {self.ad_len[k]} words.\n"
                )

            prompt += (
                "Ensure the ad integration is seamless, fits the conversational context, and that ad information is clearly visible.\n"
                "Response example: [Advertiser]'s [ad content compressed to the specified length], and also consider [Advertiser]'s [ad content compressed to the specified length],...\n"
            )

        return self._generate_with_spans(prompt, "en")

    def ad_gen_chinese_GA(self):
        return self._source_reply(english=False, with_base=False)

    def ad_gen_chinese_SQA(self):
        return self._sqa_reply(english=False)

    def ad_gen_english_GA(self):
        return self._source_reply(english=True, with_base=False)

    def ad_gen_english_SQA(self):
        return self._sqa_reply(english=True)

    def _sqa_reply(self, english):
        capacity = self.q
        self.q = 1
        try:
            return self.ad_gen_english() if english else self.ad_gen_chinese()
        finally:
            self.q = capacity

    def _source_reply(self, english, with_base):
        parts = [self.basic_reply] if with_base else []
        for k in range(self.q):
            if self.ad_len[k] <= 0:
                continue
            i = int(self.sigma[k])
            source = self.content[i]
            unit = "words" if english else "characters"
            text = (source if self.auc_alg is GA and count_text(source, unit) <= self.ad_len[k]
                    else prefix_text(source, self.ad_len[k], unit))
            marked = f'<ad id="{i}">{text}</ad>'
            parts.append(marked + (' ' if english else '') if with_base
                         else f'{k + 1}. {marked};\n')
        return self._record_reply(''.join(parts))

    def satisfaction(self):
        if self._span_reply != self.compressed_ad:
            self._record_reply(self.compressed_ad)
        pairs = [(self.compressed_ad, self.basic_reply)]
        pairs.extend((record["text"], self.content[record["advertiser_index"]])
                     for record in self.ad_spans)
        similarities = self.scorer.similarities(pairs)
        self.user_satis = float(similarities[0])
        self.ad_satis = float(np.mean(similarities[1:])) if self.ad_spans else 0.0
        for record, cosine in zip(self.ad_spans, similarities[1:]):
            record["cosine"] = float(cosine)
        self.scoring = {**self.scorer.metadata(), "ad_span_policy": "internal_ad_id_markers_v1",
                        "missing_or_invalid_span_score": 0.0,
                        "Su_scope": "complete_clean_reply_vs_complete_base",
                        "Sa_scope": "mean_over_q_selected_advertisers",
                        "Sa_denominator": len(self.ad_spans)}
        if self.auc_alg is GA:
            self.scoring.update(Sa_scope="mean_over_displayed_GA_advertisers",
                                empty_display_Sa_policy="zero_when_no_ads")
        self.score = self.theta_1 * self.user_satis + (1 - self.theta_1) * self.ad_satis + self.theta_2 * self.SW

    def reply(self, query):
        raise NotImplementedError

    def auction(self):
        len_content = [count_text(ads, self.length_unit) for ads in self.content]
        self.sigma, self.prom, self.payments, self.utilities, self.SW = self.auc_alg(self.b, self.pctr, self.pos_norm,
                                                                                     self.q, self.constant, len_content,
                                                                                     self.max_len)
        self.ad_len = [int(self.max_len * p) for p in self.prom]
        if self.auc_alg is GA:
            # Recover integer allocations without floor-roundoff.
            self.ad_len = [int(round(self.max_len * p)) for p in self.prom]

    def auction_DSIC(self, beta, s):
        len_content = [count_text(ads, self.length_unit) for ads in self.content]
        self.sigma, self.prom, self.payments, self.utilities, self.SW = self.auc_alg(self.b, self.pctr, self.pos_norm,
                                                                                     self.q, self.constant, len_content,
                                                                                     self.max_len, beta, s)
        self.ad_len = [int(self.max_len * p) for p in self.prom]

