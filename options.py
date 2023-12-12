import argparse


class options_classification(object):

    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description='Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments.')

        ## Run from config file
        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')

        # gpt4ts
        self.parser.add_argument('--patch_size', type=int, default=64, help='patch_size')
        self.parser.add_argument('--stride', type=int, default=64, help='stride')

        ## Run from command-line arguments
        # I/O
        self.parser.add_argument('--output_dir', default='./output',
                                 help='Root output directory. Must exist. Time-stamped directories will be created inside.')
        self.parser.add_argument('--data_dir', default='./datasets_classification',
                                 help='Data directory')
        self.parser.add_argument('--load_model',
                                 help='Path to pre-trained model.')
        self.parser.add_argument('--resume', action='store_true',
                                 help='If set, will load `starting_epoch` and state of optimizer, besides model weights.')
        self.parser.add_argument('--change_output', action='store_true',
                                 help='Whether the loaded model will be fine-tuned on a different task (necessitating a different output layer)')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='If set, will save model weights (and optimizer state) for every epoch; otherwise just latest')
        self.parser.add_argument('--name', dest='experiment_name', default='',
                                 help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
        self.parser.add_argument('--comment', type=str, default='', help='A comment/description of the experiment')
        self.parser.add_argument('--no_timestamp', action='store_true',
                                 help='If set, a timestamp will not be appended to the output directory name')
        self.parser.add_argument('--records_file', default='./records.xls',
                                 help='Excel file keeping all records of experiments')
        # system
        self.parser.add_argument('--console', action='store_true',
                                 help="Optimize printout for console output; otherwise for file")
        self.parser.add_argument('--print_interval', type=int, default=1,
                                 help='Print batch info every this many batches')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='GPU index, -1 for CPU')
        self.parser.add_argument('--n_proc', type=int, default=-1,
                                 help='Number of processes for data loading/preprocessing. By default, equals num. of available cores.')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed',
                                 help='Seed used for splitting sets. None by default, set to an integer for reproducibility')
        # dataset
        self.parser.add_argument('--limit_size', type=float, default=None,
                                 help="Limit  dataset to specified smaller random sample, e.g. for rapid debugging purposes. "
                                      "If in [0,1], it will be interpreted as a proportion of the dataset, "
                                      "otherwise as an integer absolute number of samples")
        self.parser.add_argument('--test_only', choices={'testset', 'fold_transduction'},
                                 help='If set, no training will take place; instead, trained model will be loaded and evaluated on test set')
        self.parser.add_argument('--data_class', type=str, default='tsra',
                                 help="Which type of data should be processed.")
        self.parser.add_argument('--labels', type=str,
                                 help="In case a dataset contains several labels (multi-task), "
                                      "which type of labels should be used in regression or classification, i.e. name of column(s).")
        self.parser.add_argument('--test_from',
                                 help='If given, will read test IDs from specified text file containing sample IDs one in each row')
        self.parser.add_argument('--test_ratio', type=float, default=0,
                                 help="Set aside this proportion of the dataset as a test set")
        self.parser.add_argument('--val_ratio', type=float, default=0.2,
                                 help="Proportion of the dataset to be used as a validation set")
        self.parser.add_argument('--pattern', type=str,
                                 help='Regex pattern used to select files contained in `data_dir`. If None, all data will be used.')
        self.parser.add_argument('--val_pattern', type=str,
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the validation set.
                            If None, a positive `val_ratio` will be used to reserve part of the common data set.""")
        self.parser.add_argument('--test_pattern', type=str,
                                 help="""Regex pattern used to select files contained in `data_dir` exclusively for the test set.
                            If None, `test_ratio`, if specified, will be used to reserve part of the common data set.""")
        self.parser.add_argument('--normalization',
                                 choices={'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'},
                                 default='standardization',
                                 help='If specified, will apply normalization on the input features of a dataset.')
        self.parser.add_argument('--norm_from',
                                 help="""If given, will read normalization values (e.g. mean, std, min, max) from specified pickle file.
                            The columns correspond to features, rows correspond to mean, std or min, max.""")
        self.parser.add_argument('--subsample_factor', type=int,
                                 help='Sub-sampling factor used for long sequences: keep every kth sample')
        # Training process
        self.parser.add_argument('--task', choices={"imputation", "transduction", "classification", "regression"},
                                 default="imputation",
                                 help=("Training objective/task: imputation of masked values,\n"
                                       "                          transduction of features to other features,\n"
                                       "                          classification of entire time series,\n"
                                       "                          regression of scalar(s) for entire time series"))
        self.parser.add_argument('--masking_ratio', type=float, default=0.15,
                                 help='Imputation: mask this proportion of each variable')
        self.parser.add_argument('--mean_mask_length', type=float, default=3,
                                 help="Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.")
        self.parser.add_argument('--mask_mode', choices={'separate', 'concurrent'}, default='separate',
                                 help=("Imputation: whether each variable should be masked separately "
                                       "or all variables at a certain positions should be masked concurrently"))
        self.parser.add_argument('--mask_distribution', choices={'geometric', 'bernoulli'}, default='geometric',
                                 help=("Imputation: whether each mask sequence element is sampled independently at random"
                                       "or whether sampling follows a markov chain (stateful), resulting in "
                                       "geometric distributions of masked squences of a desired mean_mask_length"))
        self.parser.add_argument('--exclude_feats', type=str, default=None,
                                 help='Imputation: Comma separated string of indices corresponding to features to be excluded from masking')
        self.parser.add_argument('--mask_feats', type=str, default='0, 1',
                                 help='Transduction: Comma separated string of indices corresponding to features to be masked')
        self.parser.add_argument('--start_hint', type=float, default=0.0,
                                 help='Transduction: proportion at the beginning of time series which will not be masked')
        self.parser.add_argument('--end_hint', type=float, default=0.0,
                                 help='Transduction: proportion at the end of time series which will not be masked')
        self.parser.add_argument('--harden', action='store_true',
                                 help='Makes training objective progressively harder, by masking more of the input')

        self.parser.add_argument('--epochs', type=int, default=400,
                                 help='Number of training epochs')
        self.parser.add_argument('--val_interval', type=int, default=2,
                                 help='Evaluate on validation set every this many epochs. Must be >= 1.')
        self.parser.add_argument('--optimizer', choices={"Adam", "RAdam"}, default="Adam", help="Optimizer")
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='learning rate (default holds for batch size 64)')
        self.parser.add_argument('--lr_step', type=str, default='1000000',
                                 help='Comma separated string of epochs when to reduce learning rate by a factor of 10.'
                                      ' The default is a large value, meaning that the learning rate will not change.')
        self.parser.add_argument('--lr_factor', type=str, default='0.1',
                                 help=("Comma separated string of multiplicative factors to be applied to lr "
                                       "at corresponding steps specified in `lr_step`. If a single value is provided, "
                                       "it will be replicated to match the number of steps in `lr_step`."))
        self.parser.add_argument('--batch_size', type=int, default=64,
                                 help='Training batch size')
        self.parser.add_argument('--l2_reg', type=float, default=0,
                                 help='L2 weight regularization parameter')
        self.parser.add_argument('--global_reg', action='store_true',
                                 help='If set, L2 regularization will be applied to all weights instead of only the output layer')
        self.parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='loss',
                                 help='Metric used for defining best epoch')
        self.parser.add_argument('--freeze', action='store_true',
                                 help='If set, freezes all layer parameters except for the output layer. Also removes dropout except before the output layer')

        # gpt2 model
        self.parser.add_argument('--model', choices={"transformer", "LINEAR"}, default="transformer",
                                 help="Model class")
        self.parser.add_argument('--max_seq_len', type=int,
                                 help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""")
        self.parser.add_argument('--data_window_len', type=int,
                                 help="""Used instead of the `max_seq_len`, when the data samples must be
                                 segmented into windows. Determines maximum input sequence length 
                                 (size of transformer layers).""")
        self.parser.add_argument('--d_model', type=int, default=64,
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--dim_feedforward', type=int, default=256,
                                 help='Dimension of dense feedforward part of transformer layer')
        self.parser.add_argument('--num_heads', type=int, default=8,
                                 help='Number of multi-headed attention heads')
        self.parser.add_argument('--num_layers', type=int, default=3,
                                 help='Number of transformer encoder layers (blocks)')
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help='Dropout applied to most transformer encoder layers')
        self.parser.add_argument('--pos_encoding', choices={'fixed', 'learnable'}, default='fixed',
                                 help='Internal dimension of transformer embeddings')
        self.parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                                 help='Activation to be used in transformer encoder')
        self.parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                                 help='Normalization layer to be used internally in transformer encoder')

        # text prototype
        self.parser.add_argument('--type_of_prototype', type=str, default='representative',
                            help='Provide or select the prototypes. ([provide text list] or random or representative)')
        self.parser.add_argument('--number_of_prototype', type=int, default=10, metavar='Number',
                            help='Number of prototype to select')

        # encoder
        self.parser.add_argument('--load_encoder', action='store_true', default=False,
                            help='activate to load the estimator instead of ' +
                                 'training it')
        self.parser.add_argument('--fit_encoder_classifier', action='store_true', default=False,
                            help='if not supervised, activate to load the ' +
                                 'model and retrain the classifier')
        self.parser.add_argument('--encoder_save_path', type=str, metavar='PATH', default='./encoders/saved_models',
                            help='path where the estimator is/should be saved')



    def parse(self):

        args = self.parser.parse_args()

        args.lr_step = [int(i) for i in args.lr_step.split(',')]
        args.lr_factor = [float(i) for i in args.lr_factor.split(',')]
        if (len(args.lr_step) > 1) and (len(args.lr_factor) == 1):
            args.lr_factor = len(args.lr_step) * args.lr_factor  # replicate
        assert len(args.lr_step) == len(
            args.lr_factor), "You must specify as many values in `lr_step` as in `lr_factors`"

        if args.exclude_feats is not None:
            args.exclude_feats = [int(i) for i in args.exclude_feats.split(',')]
        args.mask_feats = [int(i) for i in args.mask_feats.split(',')]

        if args.val_pattern is not None:
            args.val_ratio = 0
            args.test_ratio = 0

        return args


class options_forecasting(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser(description='gpt4ts long-term forecasting')

        self.parser.add_argument('--config', dest='config_filepath',
                                 help='Configuration .json file (optional). Overwrites existing command-line args!')

        self.parser.add_argument('--model_id', type=str, required=True, default='test')
        self.parser.add_argument('--output_dir', type=str, default='./output')
        self.parser.add_argument('--name', dest='experiment_name', default='',
                                 help='A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp')
        self.parser.add_argument('--no_timestamp', action='store_true',
                                 help='If set, a timestamp will not be appended to the output directory name')
        self.parser.add_argument('--seed',
                                 help='Seed used for splitting sets. None by default, set to an integer for reproducibility')
        self.parser.add_argument('--gpu', type=str, default='0',
                                 help='GPU index, -1 for CPU')

        self.parser.add_argument('--root_path', type=str, default='./dataset_forecasting/traffic/')
        self.parser.add_argument('--data_path', type=str, default='traffic.csv')
        self.parser.add_argument('--data', type=str, default='custom')
        self.parser.add_argument('--features', type=str, default='M')
        self.parser.add_argument('--freq', type=int, default=0)
        self.parser.add_argument('--target', type=str, default='OT')
        self.parser.add_argument('--embed', type=str, default='timeF')
        self.parser.add_argument('--percent', type=int, default=100)

        self.parser.add_argument('--seq_len', type=int, default=512)
        self.parser.add_argument('--pred_len', type=int, default=96)
        self.parser.add_argument('--label_len', type=int, default=48)

        self.parser.add_argument('--decay_fac', type=float, default=0.75)
        self. parser.add_argument('--learning_rate', type=float, default=0.001)
        self.parser.add_argument('--batch_size', type=int, default=512)
        self.parser.add_argument('--num_workers', type=int, default=10)
        self.parser.add_argument('--train_epochs', type=int, default=10)
        self.parser.add_argument('--lradj', type=str, default='type1')
        self.parser.add_argument('--patience', type=int, default=3)

        self.parser.add_argument('--gpt_layers', type=int, default=6)
        self.parser.add_argument('--is_gpt', type=int, default=1)
        self.parser.add_argument('--e_layers', type=int, default=3)
        self.parser.add_argument('--d_model', type=int, default=768)
        self.parser.add_argument('--n_heads', type=int, default=4)
        self.parser.add_argument('--d_ff', type=int, default=768)
        self.parser.add_argument('--dropout', type=float, default=0.2)
        self.parser.add_argument('--enc_in', type=int, default=862)
        self.parser.add_argument('--c_out', type=int, default=862)
        self.parser.add_argument('--patch_size', type=int, default=16)
        self.parser.add_argument('--kernel_size', type=int, default=25)

        self.parser.add_argument('--loss_func', type=str, default='mse')
        self.parser.add_argument('--pretrain', type=int, default=1)
        self.parser.add_argument('--freeze', type=int, default=1)
        self.parser.add_argument('--model', type=str, default='gpt4ts')
        self.parser.add_argument('--stride', type=int, default=8)
        self.parser.add_argument('--max_len', type=int, default=-1)
        self.parser.add_argument('--hid_dim', type=int, default=16)
        self.parser.add_argument('--tmax', type=int, default=10)

        self.parser.add_argument('--itr', type=int, default=3)
        self.parser.add_argument('--cos', type=int, default=1)


        # text prototype
        self.parser.add_argument('--type_of_prototype', type=str, default='representative',
                            help='Provide or select the prototypes. ([provide text list] or random or representative)')
        self.parser.add_argument('--number_of_prototype', type=int, default=10, metavar='Number',
                            help='Number of prototype to select')

        # encoder
        self.parser.add_argument('--load_encoder', action='store_true', default=False,
                            help='activate to load the estimator instead of ' +
                                 'training it')
        self.parser.add_argument('--fit_encoder_classifier', action='store_true', default=False,
                            help='if not supervised, activate to load the ' +
                                 'model and retrain the classifier')
        self.parser.add_argument('--encoder_save_path', type=str, metavar='PATH', default='./encoders/saved_models',
                            help='path where the estimator is/should be saved')



    def parse(self):

        args = self.parser.parse_args()

        return args
