from src.schedules import build_betas_schedule


class TrainConfig:
    def __init__(self, args):
        self.total_epochs = args.total_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.beta = args.beta
        self.schedule = args.schedule

        self.seed = args.seed
        self.checkpoint_dir = args.checkpoint_dir
        self.save_freq = args.save_freq
        self.eval_freq = args.eval_freq

        self.beta_schedule = build_betas_schedule(self.schedule, self.beta, self.total_epochs)

    def get_beta(self, epoch):
        return self.beta_schedule[epoch]


class HyperConfig:
    def __init__(self, args):
        self.hyper_type = args.hyper_type
        self.block_type = args.block_type
        self.include_output_linear = args.include_output_linear
        self.include_sigmoid_activation = args.include_sigmoid_activation

        self.sample_type = args.sample_type
        self.sample_range = args.sample_range
