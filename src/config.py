class HyperConfig:
    def __init__(self, args):
        self.hyper_type = args.hyper_type
        self.block_type = args.block_type
        self.include_output_linear = args.include_output_linear
        self.include_sigmoid_activation = args.include_sigmoid_activation

        self.sample_type = args.sample_type
        self.sample_range = args.sample_range
