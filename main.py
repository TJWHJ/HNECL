import argparse
from logging import getLogger

import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color
from HNECL import HNECL
from trainer import HNECLTrainer
from time import time
import os
import pickle


def run_single_model(args):
    # configurations initialization
    config = Config(
        model=HNECL,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    t1 = time()
    print('create_dataset...')
    file = config['filterd_datset_path']
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
    else:
        dataset = create_dataset(config)

    logger.info(dataset)
    print('done.', time() - t1)

    # dataset splitting
    t2 = time()
    print('data_preparation...')
    train_data, valid_data, test_data = data_preparation(config, dataset)
    print('done.', time() - t2)

    # model loading and
    model = HNECL(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = HNECLTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon-books',
                        help='The datasets can be: ml-1m, yelp, amazon-books, gowalla-merged, alibaba.')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--local_rank', default=-1, type=int)  ###
    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
    ]
    if args.dataset in ['ml-1m', 'yelp', 'amazon-books', 'gowalla-merged', 'alibaba']:
        args.config_file_list.append(f'properties/{args.dataset}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)

    run_single_model(args)
