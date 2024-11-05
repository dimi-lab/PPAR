
import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Set, Union, NewType, Dict, Optional 
import logging
import logging
from collections import defaultdict
import time
from tqdm import tqdm

def main(HPO_list, global_hpo, model, graph_path, prob_file, total_genes):
    logger = setup_logger()
    glbal_HPO_list = create_HPO_list(global_hpo)
    querry_HPO_list = validate_input(HPO_list, glbal_HPO_list)
    predict_rank_graph(model, HPO_list, graph_path, prob_file, total_genes )
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



def find_top_common_ancestor_genes(phenotype_list, graph, phenotype_df, top_n):
    gene_weight_sum_dict = defaultdict(float)
    gene_count_dict = defaultdict(int)
    
    for phenotype in phenotype_list:
        if phenotype not in graph:
            print(f"Phenotype {phenotype} not found in the graph. Ignoring.")
            continue
        connected_genes = {neighbor for neighbor in graph.neighbors(phenotype) if 'GO' not in neighbor}
        phenotype_weight_row = phenotype_df.loc[phenotype_df['HPO'] == phenotype]
        
        if not phenotype_weight_row.empty:
            phenotype_weight = phenotype_weight_row['PROBABILITY'].values[0]
        else:
            phenotype_weight = 1 
        for gene in connected_genes:
            gene_weight_sum_dict[gene] += 1 - phenotype_weight
            gene_count_dict[gene] += 1
    gene_score_dict = {gene: gene_count_dict[gene] * gene_weight_sum_dict[gene] if gene_weight_sum_dict[gene] > 0 else 0 
                       for gene in gene_count_dict}
    sorted_genes = sorted(gene_score_dict.items(), key=lambda x: x[1], reverse=True)
    top_genes = [(gene, score, gene_count_dict[gene]) for gene, score in sorted_genes[:top_n]]
    return top_genes

    
def predict_rank_graph(model: str, HPO: list, graph: str, prob: str, k: int):
    G = nx.read_graphml(graph)
    rank_dict = []
    model_df = pd.read_csv(model)
    model_df = model_df.set_index('Gene')
    prob_df = pd.read_csv(prob)
    HPO_list = HPO
    result_df = model_df[HPO_list]
    total_count = len(HPO_list)
    filter_HPO = list(set(HPO_list).intersection(model_df.columns))    
    if len(filter_HPO) == 0:
        print(f"No matching HPOs found for gene {gene}. Skipping...")

    top_genes = find_top_common_ancestor_genes(filter_HPO, G, prob_df, top_n=2000)
    top_genes_df = pd.DataFrame(top_genes, columns=['Gene', 'Score', 'Phenotype_Count'])
    result_df = pd.DataFrame(model_df[filter_HPO])
    result_max_df = pd.DataFrame(result_df.max(axis=1), columns=['MaxValue']).reset_index()
    result_mean_df = pd.DataFrame(result_df.mean(axis=1), columns=['MeanValue']).reset_index()

    result_max_df = result_max_df.merge(top_genes_df[['Gene', 'Score']], on='Gene', how='left')
    result_mean_df = result_mean_df.merge(top_genes_df[['Gene', 'Score']], on='Gene', how='left')
            
    std_dev_max_value = result_max_df['MaxValue'].std()
    std_dev_mean_value = result_mean_df['MeanValue'].std()

    result_max_df['MaxValue'] = np.where(result_max_df['Score'].isna(), 
                                         result_max_df['MaxValue'], 
                                         result_max_df['MaxValue'] + (result_max_df['Score'] * std_dev_max_value))

    result_max_df = result_max_df.sort_values(by='MaxValue',
                                              ascending=False).reset_index(drop=True)
    top_k_max = result_max_df['Gene'].head(k).tolist()
    with open('PPAR_results_top.tsv', 'w+') as fout:
        for i, gene in enumerate(top_k_max):
            rank = i +1
            fout.write(f"{rank}\t{gene}\n")

        

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
    parser.add_argument('-i', dest='HPO_list', nargs='+', required=True,
                        help='list of input HPO terms')    
    parser.add_argument('-m', dest='model', default='../data/PPAR_cosine_data.csv',
                        help='path to static PPAR matrix ')
    parser.add_argument('-g', dest='graph_path', default='../data/RPAR_gene_phenotype_graph.graphml', help='path to the gene-HPO graph') 
    parser.add_argument('-p', dest='prob_file', default='../data/hpo_probability_custom.csv',
                        help='List of all probabilities for the HPO terms')
    parser.add_argument('-k', dest='total_genes', default=100,
                        help='total output genes', type=int)
    parser.add_argument('-gh', dest='global_hpo', default='../data/global_HPO.csv',
                        help='comprehensive HPO list the model knows')                            
    args = parser.parse_args()
    main(args.HPO_list, args.global_hpo, args.model, args.graph_path,
         args.prob_file, args.total_genes)
