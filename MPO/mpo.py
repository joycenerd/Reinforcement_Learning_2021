from options import args


class MPO(object):
    def __init__(self):
        self.env = args.env
        self.eps = args.eps
        self.eps_mu = args.eps_mu
        self.eps_sigma = args.eps_sigma
        self.lr = args.lr
        self.alpha = args.alpha
        self.epochs = args.epochs
        self.epoch_length = args.epoch_length
        self.lagrange_iter = args.lagrange_iter
        self.batch_size = args.batch_size
        self.sample_epochs = args.sample_epochs
        self.add_act = args.add_act
        self.policy_layers = args.policy_layers
        self.Q_layers = args.Q_layers
        self.log = args.log
        self.log_dir = args.log_dir
        self.render = args.render
