
################################################################


MODEL :        RPAR

DEVELOPER:  Rohan David Gnanaolivu

Email: gnanaolivu.Rohandavid@mayo.edu


The Rank order prediction of genes for a given set of phenotypes for Mendelian and rare disease framework (RPAR) prioritizes candidate diagnostic-relevant genes for genetics conditions, leveraging a large-scale clinical knowledge graph (CKG) consisting of 19,405,058 nodes and 217,341,612 edges as relationships, curated from 24 databases and ten ontology data sources such as the Human Phenotype Ontology.  The CKG is stored in a Neo4j graph data platform. The developed framework includes (i) prediction of reliable knowledge relationships using a graph embedding approach FastRP (Fast Random Projection), and (ii) gene-phenotype relevance ranking according to the Cosine similarity of gene and phenotype nodes in CKG, using the predictions and information content for each phenotype as weights. 

#######################################################################



This version is a static version of the gene and phenotype embeddings


# Model methods

1. Link analysis method: L2
2. Link prediction: Multiple Layer Perceptron (MLP)
3. Similarity: Bayesian
4. Information content: CKG connectivity


# Model files
Model result path: /research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/RPAR_bayes_L2_MLP.csv
All HPO list: /research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/Global_HPO.list
All gene list: /research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/Global_genes.list


#########################################################################

# code

  RPAR_run_model.py


# python package requirements
  1. pandas
  2. numpy
  3. typing
  4. os
  5. collections
  6. logging
  7. tqdm
  8. time

# usage
  python RPAR_run_model.py -h
usage: RPAR_run_model.py [-h] -p HPO_LIST [HPO_LIST ...] [-m MODEL]
                         [-g GLOBAL_HPO]

optional arguments:
  -h, --help            show this help message and exit
  -p HPO_LIST [HPO_LIST ...]
                        HPO terms to query the RPAR model
  -m MODEL              path to static model result (Default: /research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/RPAR_bayes_L2_MLP.csv)
  -g GLOBAL_HPO         List of all HPO terms in the training data (Default: /research/bsi/projects/PI/tertiary/Klee_Eric_mrl2075/s212354.RadiaNT/MultiomicsSummaries/Rohan/RPAR/results/Global_HPO.list)

# OUTPUT
1. RPAR_results.top100max.tsv (uses the ranks of the best HPO item in the list as the output)
2. RPAR_results.top100mean.tsv (takes the mean ranks of all HPO items in the list)

# usage example
python RPAR_run_model.py -p HP:0011403 HP:0001274 HP:0000248 HP:0011304 HP:0000175 HP:0001305 HP:0001631 HP:0000494 HP:0002280 HP:0000577 HP:0001263 HP:0000564 HP:0000368 HP:0000303 HP:0000410 HP:0000543 HP:0000767 HP:0001773 HP:0100874

Note: the -p flag takes in a list of HPO terms separated by spaces

##################################################################################

# OUTPUT

the output RPAR_results.top100max.tsv and RPAR_results.top100mean.tsv, contains two columns
column1: Rank
column2: Gene

