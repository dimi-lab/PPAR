#!/research/bsi/projects/PI/tertiary/Couch_Fergus_coucf/s123456.general_utility/python_virtual/bin/python
import pandas as pd
import numpy as np
from typing import List, Set, Union, NewType, Dict, Optional 
import logging
import logging
from collections import defaultdict
import time
from tqdm import tqdm

def main(HPO_list, model, global_HPO):
    logger = setup_logger()
    glbal_HPO_list = create_HPO_list(global_HPO)
    querry_HPO_list = validate_input(HPO_list, glbal_HPO_list)
    predict_rank(model, querry_HPO_list)
#    print(top100max)
    
def setup_logger(log_file='RPAR_log.txt'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def log(message, level='info'):
    logger = logging.getLogger()
    if level.lower() == 'debug':
        logger.debug(message)
    elif level.lower() == 'info':
        logger.info(message)
    elif level.lower() == 'warning':
        logger.warning(message)
    elif level.lower() == 'error':
        logger.error(message)
    elif level.lower() == 'critical':
        logger.critical(message)
    else:
        raise ValueError("Invalid log level")


def predict_rank(model: str, HPO: List, ):
    """ interate through the querry list of HPO terms

    Parameters
    ----------
    model: Save baysian model results with HPO and genes
    
    HPO: this is list of querry HPO terms
    
    Returns
    ---------
    Rank order list of genes
    """

    rank_dict = defaultdict(list)
    HPO_list = HPO
    df = pd.read_csv(model)
    df = df.set_index('Gene')
    result_df = df[HPO_list]
    total_count = len(HPO_list)
    result_max_df = pd.DataFrame(result_df.max(axis=1),
                                 columns=['MaxValue']).sort_values(by='MaxValue', ascending=False).reset_index()
    result_mean_df = pd.DataFrame(result_df.mean(axis=1),
                                  columns=['MeanValue']).apply(lambda x: x/len(HPO_list)).sort_values(by='MeanValue',
                                                                                                         ascending=False).reset_index()
    top_100_max = result_max_df['Gene'].head(100).tolist()
    top_100_mean = result_mean_df['Gene'].head(100).tolist()
    with open('RPAR_results.top100max.tsv', 'w+') as fout:
        for i, j in enumerate(top_100_max):
            rank = i +1
            fout.write(str(rank) + '\t' +j + '\n')
    with open('RPAR_results.top100mean.tsv','w+') as fout:
        for i, j in enumerate(top_100_mean):
            rank = i +1
            fout.write(str(rank) + '\t' +j + '\n')       
#    return top_50_max, top_50_mean


def create_HPO_list(valid_HPO: str):
    valid_hpo = []
    with open(valid_HPO) as fin:
        for raw_line in fin:
            line = raw_line.strip()
            valid_hpo.append(line)
    return valid_hpo
    
def validate_input(HPO:List, valid_HPO: List):
    querry_list = []
    for i in tqdm(HPO, desc='validating HPO terms and generating Gene list'):
        if i in valid_HPO:
            querry_list.append(i)
        else:
            log('the HPO term is not in the training data {0}'.format(i), level='error')
    return querry_list


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', dest='HPO_list',  required=True,nargs="+",
                        help='HPO terms to query the RPAR model')
    parser.add_argument('-m', dest='model', default='/research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/RPAR_bayes_L2_MLP.csv',
                        help='path to static model result')
    parser.add_argument('-g', dest='global_HPO', default='/research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/Global_HPO.list',
                        help='List of all HPO terms in the training data')    
    args = parser.parse_args()
    main(args.HPO_list, args.model, args.global_HPO)
