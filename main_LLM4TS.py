import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
import os
import sys
import time
import pickle
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project classification modules
from options import options_classification
from running import setup, pipeline_factory, validate, check_progress, NEG_METRICS
from utils import utils
from datasets_classification.data import data_factory, Normalizer
from datasets_classification.datasplit import split_dataset
from models.gpt4ts import gpt4ts_classification
from models.loss import get_loss_module
from optimizers import get_optimizer
from losses import text_prototype

# Project forecasting modules
from options import options_forecasting
from datasets_forecasting.data_factory import data_provider
from utils.utils import EarlyStopping, adjust_learning_rate, vali, test
from models.gpt4ts import gpt4ts_forecasting


def main_classification(config):
    total_epoch_time = 0
    total_start_time = time.time()

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)
    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    # Device info
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))


# Select text prototype -------------

    logger.info("Select text prototype ...")
    text_prototype_file='./losses/text_prototype_'+config['type_of_prototype']+'.pt'
    if os.path.exists(text_prototype_file):
        prototype_embeddings=torch.load(text_prototype_file)
        prototype_size=prototype_embeddings.size()
    else:
        prototype_embeddings,prototype_size=text_prototype.select_prototype(model_dir='./models/gpt2',prototype_dir='./losses',provide=config['type_of_prototype'],number_of_prototype=config['number_of_prototype'])
    logger.info("{} prototype are selected, their dimension is {}".format(prototype_size[0],prototype_size[1]))



# Build data -------------

    logger.info("Loading and preprocessing data ...")
    data_class = data_factory[config['data_class']] # the name of dataset
    my_data = data_class(config['data_dir'], pattern=config['pattern'], n_proc=config['n_proc'],
                         limit_size=config['limit_size'], config=config)

    if config['task'] == 'classification':
        validation_method = 'StratifiedShuffleSplit'
        labels = my_data.labels_df.values.flatten()
    else:
        validation_method = 'ShuffleSplit'
        labels = None

    # Split dataset
    test_data = my_data
    test_indices = None  # will be converted to empty list in `split_dataset`, if also test_set_ratio == 0
    val_data = my_data
    val_indices = []
    if config['test_pattern']:  # used if test data come from different files / file patterns
        test_data = data_class(config['data_dir'], pattern=config['test_pattern'], n_proc=-1, config=config)
        test_indices = test_data.all_IDs
    if config[
        'test_from']:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`. Can work together with `test_pattern`
        test_indices = list(set([line.rstrip() for line in open(config['test_from']).readlines()]))
        try:
            test_indices = [int(ind) for ind in test_indices]  # integer indices
        except ValueError:
            pass  # in case indices are non-integers
        logger.info("Loaded {} test IDs from file: '{}'".format(len(test_indices), config['test_from']))
    if config['val_pattern']:  # used if val data come from different files / file patterns
        val_data = data_class(config['data_dir'], pattern=config['val_pattern'], n_proc=-1, config=config)
        val_indices = val_data.all_IDs

    # Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
    # Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
    if config['val_ratio'] > 0:
        train_indices, val_indices, test_indices = split_dataset(data_indices=my_data.all_IDs,
                                                                 validation_method=validation_method,
                                                                 n_splits=1,
                                                                 validation_ratio=config['val_ratio'],
                                                                 test_set_ratio=config['test_ratio'],
                                                                 # used only if test_indices not explicitly specified
                                                                 test_indices=test_indices,
                                                                 random_seed=1337,
                                                                 labels=labels)
        train_indices = train_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
        val_indices = val_indices[0]  # `split_dataset` returns a list of indices *per fold/split*
    else:
        train_indices = my_data.all_IDs
        if test_indices is None:
            test_indices = []

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_indices),
                       'val_indices': list(val_indices),
                       'test_indices': list(test_indices)}, f, indent=4)


# build LLM -----------------

    # Pre-process features
    normalizer = None
    if config['norm_from']:
        with open(config['norm_from'], 'rb') as f:
            norm_dict = pickle.load(f)
        normalizer = Normalizer(**norm_dict)
    elif config['normalization'] is not None:
        normalizer = Normalizer(config['normalization'])
        my_data.feature_df.loc[train_indices] = normalizer.normalize(my_data.feature_df.loc[train_indices])
        if not config['normalization'].startswith('per_sample'):
            # get normalizing values from training set and store for future use
            norm_dict = normalizer.__dict__
            with open(os.path.join(config['output_dir'], 'normalization.pickle'), 'wb') as f:
                pickle.dump(norm_dict, f, pickle.HIGHEST_PROTOCOL)
    if normalizer is not None:
        if len(val_indices):
            val_data.feature_df.loc[val_indices] = normalizer.normalize(val_data.feature_df.loc[val_indices])
        if len(test_indices):
            test_data.feature_df.loc[test_indices] = normalizer.normalize(test_data.feature_df.loc[test_indices])

    # Create model
    logger.info("Creating LLM model ...")
    # model = model_factory(config, my_data)
    model = gpt4ts_classification(config, my_data)

    if config['freeze']:
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Initialize optimizer

    if config['global_reg']:
        weight_decay = config['l2_reg']
        output_reg = None
    else:
        weight_decay = 0
        output_reg = config['l2_reg']

    optim_class = get_optimizer(config['optimizer'])
    optimizer = optim_class(model.parameters(), lr=config['lr'], weight_decay=weight_decay)

    start_epoch = 0
    lr_step = 0  # current step index of `lr_step`
    lr = config['lr']  # current learning step
    # Load model and optimizer state
    if config['load_model']:
        model, optimizer, start_epoch = utils.load_model(model, config['load_model'], optimizer, config['resume'],
                                                         config['change_output'],
                                                         config['lr'],
                                                         config['lr_step'],
                                                         config['lr_factor'])
    model.to(device)

    loss_module = get_loss_module(config)

    if config['test_only'] == 'testset':  # Only evaluate and skip training
        dataset_class, collate_fn, runner_class = pipeline_factory(config)
        test_dataset = dataset_class(test_data, test_indices)

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 num_workers=config['num_workers'],
                                 pin_memory=True,
                                 collate_fn=lambda x: collate_fn(x, max_len=model.max_len))
        test_evaluator = runner_class(model, test_loader, device, loss_module,
                                      print_interval=config['print_interval'], console=config['console'])
        aggr_metrics_test, per_batch_test = test_evaluator.evaluate(keep_all=True)
        print_str = 'Test Summary: '
        for k, v in aggr_metrics_test.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        return

    # Initialize data generators
    dataset_class, collate_fn, runner_class = pipeline_factory(config)
    val_dataset = dataset_class(val_data, val_indices)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    train_dataset = dataset_class(my_data, train_indices)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers'],
                              pin_memory=True,
                              collate_fn=lambda x: collate_fn(x, max_len=model.max_len))

    trainer = runner_class(model, train_loader, device, loss_module, optimizer, l2_reg=output_reg,
                           print_interval=config['print_interval'], console=config['console'])
    val_evaluator = runner_class(model, val_loader, device, loss_module,
                                 print_interval=config['print_interval'], console=config['console'])

    tensorboard_writer = SummaryWriter(config['tensorboard_dir'])

    best_value = 1e16 if config[
                             'key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
    best_metrics = {}

    # Evaluate on validation before training
    aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config, best_metrics,
                                                          best_value, epoch=0)
    metrics_names, metrics_values = zip(*aggr_metrics_val.items())
    metrics.append(list(metrics_values))

    logger.info('Starting training...')
    for epoch in tqdm(range(start_epoch + 1, config["epochs"] + 1), desc='Training Epoch', leave=False):
        mark = epoch if config['save_all'] else 'last'
        epoch_start_time = time.time()
        aggr_metrics_train = trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
        epoch_runtime = time.time() - epoch_start_time
        print()
        print_str = 'Epoch {} Training Summary: '.format(epoch)
        for k, v in aggr_metrics_train.items():
            tensorboard_writer.add_scalar('{}/train'.format(k), v, epoch)
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
        total_epoch_time += epoch_runtime
        avg_epoch_time = total_epoch_time / (epoch - start_epoch)
        avg_batch_time = avg_epoch_time / len(train_loader)
        avg_sample_time = avg_epoch_time / len(train_dataset)
        logger.info(
            "Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
        logger.info("Avg batch train. time: {} seconds".format(avg_batch_time))
        logger.info("Avg sample train. time: {} seconds".format(avg_sample_time))

        # evaluate if first or last epoch or at specified interval
        if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config['val_interval'] == 0):
            aggr_metrics_val, best_metrics, best_value = validate(val_evaluator, tensorboard_writer, config,
                                                                  best_metrics, best_value, epoch)
            metrics_names, metrics_values = zip(*aggr_metrics_val.items())
            metrics.append(list(metrics_values))

        utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(mark)), epoch, model, optimizer)

        # Learning rate scheduling
        if epoch == config['lr_step'][lr_step]:
            utils.save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            lr = lr * config['lr_factor'][lr_step]
            if lr_step < len(config['lr_step']) - 1:  # so that this index does not get out of bounds
                lr_step += 1
            logger.info('Learning rate updated to: ', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # Difficulty scheduling
        if config['harden'] and check_progress(epoch):
            train_loader.dataset.update()
            val_loader.dataset.update()

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(config["output_dir"], "metrics_" + config["experiment_name"] + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(config["records_file"], config["initial_timestamp"], config["experiment_name"],
                          best_metrics, aggr_metrics_val, comment=config['comment'])

    logger.info('Best {} was {}. Other metrics: {}'.format(config['key_metric'], best_value, best_metrics))
    logger.info('All Done!')

    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

    return best_value




def main_forecasting(config):

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(config['output_dir'], 'output.log'))
    logger.addHandler(file_handler)
    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])

    # Device info
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # path = os.path.join(config['output_dir'], config['name']+'_'+time, 'checkpoints')
    # if not os.path.exists(path):
    #     os.makedirs(path)



    logger.info("Loading and preprocessing data ...")
    SEASONALITY_MAP = {
        "minutely": 1440,
        "10_minutes": 144,
        "half_hourly": 48,
        "hourly": 24,
        "daily": 7,
        "weekly": 1,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1
    }
    if config['freq'] == 0:
        config['freq'] = 'h'
    train_data, train_loader = data_provider(config, 'train')
    vali_data, vali_loader = data_provider(config, 'val')
    test_data, test_loader = data_provider(config, 'test')

    if config['freq']!= 'h':
        config['freq'] = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(config['freq']))



    time_now = time.time()
    train_steps = len(train_loader)

    logger.info("Creating LLM model ...")
    model = gpt4ts_forecasting(config)
    model=model.to(device=device)

    # Evaluate on validation before training
    logger.info("Evaluate on validation before training ...")
    test(model, test_data, test_loader, config, device)

    logger.info('Starting training ...')
    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=config['learning_rate'])

    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)
    if config['loss_func'] == 'mse':
        criterion = nn.MSELoss()
    elif config['loss_func'] == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()

            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))

        criterion = SMAPE()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=config['tmax'], eta_min=1e-8)

    for epoch in range(config['train_epochs']):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x)

            outputs = outputs[:, -config['pred_len']:, :]
            batch_y = batch_y[:, -config['pred_len']:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((config['train_epochs'] - epoch) * train_steps - i)
                logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, config, device)
        logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        if config['cos']:
            scheduler.step()
            logger.info("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, config)
        early_stopping(vali_loss, model, config['save_dir'])
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    best_model_path = os.path.join(config['save_dir'],'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, config, device)

    logger.info("mse = {:.4f}".format(mse))
    logger.info("mae= {:.4f}".format(mae))


if __name__ == '__main__':
    # args = options_classification().parse()
    # config = setup(args)
    # main_classification(config)

    # --output_dir
    # experiments
    # --comment
    # "classification from Scratch"
    # --name
    # EthanolConcentration
    # --records_file
    # Classification_records.xls
    # --data_dir
    # ./datasets_classification/EthanolConcentration
    # --data_class
    # tsra
    # --pattern
    # TRAIN
    # --val_pattern
    # TEST
    # --epochs
    # 50
    # --lr
    # 0.001
    # --patch_size
    # 8
    # --stride
    # 8
    # --optimizer
    # RAdam
    # --d_model
    # 768
    # --pos_encoding
    # learnable
    # --task
    # classification
    # --key_metric
    # accuracy
    # --gpu
    # -1



    args = options_forecasting().parse()
    config = setup(args)
    main_forecasting(config)
    #
    # --root_path
    # ./ datasets_forecasting / traffic /
    # --data_path
    # traffic.csv
    # --model_id
    # traffic
    # --name
    # traffic
    # --data
    # custom
    # --seq_len
    # 512
    # --label_len
    # 48
    # --pred_len
    # 96
    # --output_dir
    # ./ experiments
    # --gpu
    # -1




