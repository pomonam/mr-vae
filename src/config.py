from src.schedules import build_betas_schedule


def get_ns(args, name):
    if name in args:
        return getattr(args, name)
    return None


class TrainConfig:

    def __init__(self, args):
        self.total_epochs = get_ns(args, "total_epochs")
        self.lr = get_ns(args, "lr")
        self.batch_size = get_ns(args, "batch_size")
        self.beta = get_ns(args, "beta")
        self.schedule = get_ns(args, "schedule")

        self.seed = get_ns(args, "seed")
        self.checkpoint_dir = get_ns(args, "checkpoint_dir")
        self.save_freq = get_ns(args, "save_freq")
        self.eval_freq = get_ns(args, "eval_freq")

        if self.schedule is not None and self.beta is not None:
            self.beta_schedule = build_betas_schedule(self.schedule,
                                                      self.beta,
                                                      self.total_epochs)
        else:
            self.beta_schedule = None

    def get_beta(self, epoch):
        if self.beta_schedule is not None:
            return self.beta_schedule[epoch]


class HyperConfig:
    preprocess_dim = 64

    def __init__(self, args):
        self.block_type = get_ns(args, "block_type")

        self.include_sigmoid_activation = get_ns(args, "include_sigmoid_activation")
        self.include_layer_norm = get_ns(args, "include_layer_norm")
        self.include_output_layer = get_ns(args, "include_output_layer")
        self.include_shift = get_ns(args, "include_shift")

        self.include_residual_connection = get_ns(
            args, "include_residual_connection")
        self.include_chunk = get_ns(args, "include_chunk")

        self.preact_transform = get_ns(args, "preact_transform")

        self.sample_type = get_ns(args, "sample_type")
        self.preprocess_beta = get_ns(args, "preprocess_beta")
