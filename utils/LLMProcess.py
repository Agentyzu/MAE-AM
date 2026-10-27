class LLMProcess:
    """
    Initialize the LLMProcess class.

    Args:
        model_class: The model class used to initialize the language model.
        cfg: Configuration object.
        data: Data object.
        auc_alg: Auction algorithm.
    """
    def __init__(self, model_class, cfg, data, auc_alg):
        self.llm = model_class(cfg, data, auc_alg)
        self.basic_reply = None
        self.gen_reply = None
        self.result = {}

    def run(self):
        """
        Run the standard LLM process including reply generation, similarity calculation, auction,
        and reply satisfaction assessment.
        """
        # Generate basic reply
        self.basic_reply = self.llm.reply(self.llm.query)
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply

        # Calculate similarity
        self.llm.calculate_similar(self.basic_reply)

        # Auction process
        self.llm.auction()
        self.result["ad_rank"] = self.llm.sigma[:self.llm.q].tolist()
        self.result["ad_prom"] = self.llm.prom[:self.llm.q].tolist()
        self.result["ad_sw"] = self.llm.SW
        self.result["ad_rev"] = self.llm.payments.tolist()
        self.result["ad_len"] = self.llm.ad_len
        self.result["ad_utility"] = self.llm.utilities.tolist()

        # Generate final reply
        self.gen_reply = getattr(self.llm, f"ad_gen_{self.llm.language}")()

        self.llm.gen_reply = self.gen_reply
        self.result["gen_reply"] = self.gen_reply

        # Satisfaction calculation
        self.llm.satisfaction()
        self.result["user_satis"] = self.llm.user_satis
        self.result["ad_satis"] = self.llm.ad_satis
        self.result["score"] = self.llm.score

    def run_DSIC(self):
        """
        Test dominant-strategy incentive compatibility (DSIC) by varying each advertiser's reported bid.
        """
        # Generate basic reply
        self.basic_reply = self.llm.reply(self.llm.query)
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply

        # Calculate similarity
        self.llm.calculate_similar(self.basic_reply)

        betas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
        for s in range(len(self.llm.b)):
            for beta in betas:
                self.llm.auction_DSIC(beta, s)
                self.result[f"ad_rank_{beta}_{s}"] = self.llm.sigma[:self.llm.q].tolist()
                self.result[f"ad_prom_{beta}_{s}"] = self.llm.prom[:self.llm.q].tolist()
                self.result[f"ad_sw_{beta}_{s}"] = self.llm.SW
                self.result[f"ad_rev_{beta}_{s}"] = self.llm.payments.tolist()
                self.result[f"ad_len_{beta}_{s}"] = self.llm.ad_len
                self.result[f"ad_utility_{beta}_{s}"] = self.llm.utilities.tolist()


    def run_sqa(self):
        self.basic_reply = self.llm.reply(self.llm.query)
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply

        self.llm.calculate_similar(self.basic_reply)

        self.llm.auction()
        self.result["ad_rank"] = self.llm.sigma[:self.llm.q].tolist()
        self.result["ad_prom"] = self.llm.prom[:self.llm.q].tolist()
        self.result["ad_sw"] = self.llm.SW
        self.result["ad_rev"] = self.llm.payments.tolist()
        self.result["ad_len"] = self.llm.ad_len
        self.result["ad_utility"] = self.llm.utilities.tolist()

        self.gen_reply = getattr(self.llm, f"ad_gen_{self.llm.language}_SQA")()
        self.llm.gen_reply = self.gen_reply
        self.result["gen_reply"] = self.gen_reply

        self.llm.satisfaction()
        self.result["user_satis"] = self.llm.user_satis
        self.result["ad_satis"] = self.llm.ad_satis
        self.result["score"] = self.llm.score

    def run_als(self):
        self.basic_reply = self.llm.reply(self.llm.query)
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply

        self.llm.b = self.llm.generate_b()
        self.llm.auction()
        self.result["ad_rank"] = self.llm.sigma[:self.llm.q].tolist()
        self.result["ad_prom"] = self.llm.prom[:self.llm.q].tolist()
        self.result["ad_sw"] = self.llm.SW
        self.result["ad_rev"] = self.llm.payments.tolist()
        self.result["ad_len"] = self.llm.ad_len
        self.result["ad_utility"] = self.llm.utilities.tolist()

        self.gen_reply = getattr(self.llm, f"ad_gen_{self.llm.language}_ALS")()
        self.llm.gen_reply = self.gen_reply
        self.result["gen_reply"] = self.gen_reply

        self.llm.satisfaction()
        self.result["user_satis"] = self.llm.user_satis
        self.result["ad_satis"] = self.llm.ad_satis
        self.result["score"] = self.llm.score

    def run_aal(self):
        self.basic_reply = self.llm.reply(self.llm.query)
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply

        self.llm.calculate_similar(self.basic_reply)

        self.llm.auction()
        self.result["ad_rank"] = self.llm.sigma[:self.llm.q].tolist()
        self.result["ad_prom"] = self.llm.prom[:self.llm.q].tolist()
        self.result["ad_sw"] = self.llm.SW
        self.result["ad_rev"] = self.llm.payments.tolist()
        self.result["ad_len"] = self.llm.ad_len
        self.result["ad_utility"] = self.llm.utilities.tolist()

        self.gen_reply = getattr(self.llm, f"ad_gen_{self.llm.language}_ALS")()
        self.llm.gen_reply = self.gen_reply
        self.result["gen_reply"] = self.gen_reply

        self.llm.satisfaction()
        self.result["user_satis"] = self.llm.user_satis
        self.result["ad_satis"] = self.llm.ad_satis
        self.result["score"] = self.llm.score

    def run_ga(self):
        self.basic_reply = self.llm.reply(self.llm.query)
        self.llm.basic_reply = self.basic_reply
        self.result["query"] = self.llm.query
        self.result["basic_reply"] = self.basic_reply

        self.llm.calculate_similar(self.basic_reply)

        self.llm.auction()
        self.result["ad_rank"] = self.llm.sigma[:self.llm.q].tolist()
        self.result["ad_prom"] = self.llm.prom[:self.llm.q].tolist()
        self.result["ad_sw"] = self.llm.SW
        self.result["ad_rev"] = self.llm.payments.tolist()
        self.result["ad_len"] = self.llm.ad_len
        self.result["ad_utility"] = self.llm.utilities.tolist()

        self.gen_reply = getattr(self.llm, f"ad_gen_{self.llm.language}_GA")()
        self.llm.gen_reply = self.gen_reply
        self.result["gen_reply"] = self.gen_reply

        self.llm.satisfaction()
        self.result["user_satis"] = self.llm.user_satis
        self.result["ad_satis"] = self.llm.ad_satis
        self.result["score"] = self.llm.score
