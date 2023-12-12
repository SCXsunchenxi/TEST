import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
import os
import sys
import json
import torch

# Project classification modules
from options import options_classification
from running import setup, fit_encoder_classifier_parameters
from datasets_classification.dataset import load_UEA_dataset
from losses import text_prototype
from encoders import wrapper

# Project forecasting modules
from options import options_forecasting
from datasets_forecasting.data_factory import data_provider,load_forecasting_dataset
from running import fit_encoder_parameters

def main_classification(config):

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




# Build encoder-----------------

    # Prepare data
    logger.info("Loading and preprocessing data ...")
    encoder_train_data, encoder_train_labels, encoder_test_data, encoder_test_labels = load_UEA_dataset(
        'datasets_classification/UEA_arff', config['data_dir'].split('/')[-1])

    logger.info("Creating encoder model ...")
    if device == 'cuda':
        encoder_cuda=True
        encoder_gpu=torch.cuda.current_device()
    else:
        encoder_cuda=False
        encoder_gpu=-1
    if not config['load_encoder'] and not config['fit_encoder_classifier']:
        encoder_classifier = fit_encoder_classifier_parameters(text_prototype=prototype_embeddings, dataset_x=encoder_train_data, dataset_labels=encoder_train_labels, cuda=encoder_cuda, gpu=encoder_gpu, local_rank=-1,
            save_memory=True)
    else:
        encoder_classifier = wrapper.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                config['encoder_save_path'], config['data_dir'].split('/')[-1] + '_hyperparameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = encoder_cuda
        hp_dict['gpu'] = encoder_gpu
        hp_dict['local_rank'] = -1
        hp_dict['out_channels'] = prototype_embeddings.size(1)
        encoder_classifier.set_params(**hp_dict)
        encoder_classifier.load(os.path.join(config['encoder_save_path'], config['data_dir'].split('/')[-1]))

    if not config['load_encoder']:
        if config['fit_encoder_classifier']:
            encoder_classifier.fit_classifier(encoder_classifier.encode(encoder_train_data), encoder_train_labels)
        encoder_classifier.save(
            os.path.join(config['encoder_save_path'], config['data_dir'].split('/')[-1])
        )
        with open(
            os.path.join(
                config['encoder_save_path'], config['data_dir'].split('/')[-1] + '_hyperparameters.json'
            ), 'w'
        ) as fp:
            json.dump(encoder_classifier.get_params(), fp)

    print("Test accuracy: " + str(encoder_classifier.score(encoder_test_data, encoder_test_labels)))





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


# Select text prototype -------------

    logger.info("Select text prototype ...")
    text_prototype_file='./losses/text_prototype_'+config['type_of_prototype']+'.pt'
    if os.path.exists(text_prototype_file):
        prototype_embeddings=torch.load(text_prototype_file)
        prototype_size=prototype_embeddings.size()
    else:
        prototype_embeddings,prototype_size=text_prototype.select_prototype(model_dir='./models/gpt2',prototype_dir='./losses',provide=config['type_of_prototype'],number_of_prototype=config['number_of_prototype'])
    logger.info("{} prototype are selected, their dimension is {}".format(prototype_size[0],prototype_size[1]))




# Build encoder-----------------
    # Prepare data
    logger.info("Loading and preprocessing data ...")
    #
    # encoder_train_data, encoder_train_labels, encoder_test_data, encoder_test_labels = load_UEA_dataset(
    #     'datasets_classification/UEA_arff', config['data_dir'].split('/')[-1]) # 261,3,1751

    encoder_train_data=load_forecasting_dataset(data_file_path= os.path.join(config['root_path'],config['data_path']))

    logger.info("Creating encoder model ...")
    if device == 'cuda':
        encoder_cuda=True
        encoder_gpu=torch.cuda.current_device()
    else:
        encoder_cuda=False
        encoder_gpu=-1

    if not config['load_encoder']:
        encoder = fit_encoder_parameters(text_prototype=prototype_embeddings, dataset_x=encoder_train_data, cuda=encoder_cuda, gpu=encoder_gpu, local_rank=-1,
            save_memory=True)
        encoder.save_encoder(
            os.path.join(config['encoder_save_path'], config['data_dir'].split('/')[-1])
        )

    else:
        encoder = wrapper.TimeSeriesCausalCNNEncoder()
        hf = open(
            os.path.join(
                config['encoder_save_path'], config['data_dir'].split('/')[-1] + '_hyperparameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = encoder_cuda
        hp_dict['gpu'] = encoder_gpu
        hp_dict['local_rank'] = -1
        hp_dict['out_channels'] = prototype_embeddings.size(1)
        encoder.set_params(**hp_dict)
        encoder.load_encoder(os.path.join(config['encoder_save_path'], config['data_dir'].split('/')[-1]))





if __name__ == '__main__':
    # args = options_classification().parse()  # `argsparse` object
    # config = setup(args)  # configuration dictionary
    # main_classification(config)

    # --output_dir
    # experiments_encoder
    # --data_dir
    # ./datasets/EthanolConcentration
    # --d_model
    # 768
    # --gpu
    # -1

    args = options_forecasting().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main_forecasting(config)

