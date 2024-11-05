import re
import csv
from goatools import obo_parser
from goatools.anno.genetogo_reader import Gene2GoReader
import argparse
import pandas as pd
import collections

def load_go_dag(go_obo_file):
    """Load the GO DAG from an OBO file."""
    return obo_parser.GODag(go_obo_file)

def load_gene_symbol_mapping(gene_info_file):
    """Load a mapping between Gene IDs and Gene Symbols from the gene_info file."""
    gene_info_df = pd.read_csv(gene_info_file, sep='\t')
    gene_info_df['Symbol'] = gene_info_df['Symbol'].str.upper()
    gene_info_df = gene_info_df[['GeneID', 'Symbol']]
    
    # Create a dictionary mapping Gene IDs to Gene Symbols
    gene_symbol_dict = dict(zip(gene_info_df['Symbol'],gene_info_df['GeneID']))
    
    return gene_symbol_dict

def parse_custom_gene2go(gene2go_file, taxid=9606):
    """Parse a custom gene2go format and return a defaultdict mapping Gene IDs to GO terms."""
    gene2go_dict = collections.defaultdict(list)

    with open(gene2go_file, 'r') as file:
        for raw_line in file:
            line = raw_line.strip().split("\t")  # Ensure you remove any extra whitespace
            tax_id = line[0]
            gene_id = line[1]
            go_id = line[2]
            
            # Ensure tax_id matches and both gene_id and go_id are present
            if tax_id == str(taxid) and gene_id and go_id:
                gene2go_dict[gene_id].append(go_id)

    return gene2go_dict

def get_go_terms_for_gene(gene_id, gene2go_data, go_dag):
    """Get GO terms for a given gene ID."""
    go_terms_dict = collections.defaultdict(list)
    print(gene_id)
#    print(gene2go_data)
    print(gene2go_data[gene_id])
#        go_ids = gene2go_data[gene_id]  # Get associated GO IDs for the gene
#        go_terms = {go_id: go_dag[go_id].name for go_id in go_ids if go_id in go_dag}  # Retrieve GO term names
#        go_terms_dict[gene_id].append(go_ids)
#    else:
#        print(f"No GO terms found for gene {gene_id}.")
#    
#    return go_terms_dict

def save_go_terms_to_csv(gene_list, gene2go_data, go_dag, gene_symbol_dict, output_file):
    """Save gene IDs, gene symbols, and their GO terms to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Gene_ID', 'Gene_Symbol', 'GO_ID', 'GO_Term'])
        for gene_symbol in gene_list:
            # Get the corresponding Gene ID from the gene symbol
            gene_id = gene_symbol_dict.get(gene_symbol)
            if gene_id:
                go_terms = get_go_terms_for_gene(gene_id, gene2go_data, go_dag)
                gene_symbol = gene_symbol_dict.get(gene_id, "Unknown")  # Default to "Unknown" if symbol not found
                if gene_id in go_terms:
                    for go_id, go_term in go_terms[gene_id].items():
                        writer.writerow([gene_id, gene_symbol, go_id, go_term])
            else:
                print('not working')

def read_gene_list(gene_list_file):
    """Read the list of gene IDs from a file."""
    with open(gene_list_file, 'r') as file:
        gene_list = [line.strip() for line in file ]
    return gene_list


def main():
    """Main function to retrieve GO terms for a list of genes and save to a CSV file."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Retrieve GO terms for a list of gene IDs and save to CSV.")
    parser.add_argument('--gene_list', type=str,
                        help="Path to the file containing the list of gene IDs.")
    parser.add_argument('--go_obo_file', type=str, default='/go-basic.obo',
                        help="Path to the GO OBO file (e.g., go-basic.obo).")
    parser.add_argument('--gene2go_file', type=str, default='/gene2go',
                        help="Path to the gene2go file (e.g., gene2go).")
    parser.add_argument('--gene_info_file', type=str, default='/All_Data.gene_info',
                        help="Path to the gene2go file (e.g., gene2go).")    
    parser.add_argument('--output_file', type=str, help="Path to the output CSV file.")
    
    args = parser.parse_args()

    # Read gene list from file
    gene_list = read_gene_list(args.gene_list)

    # Load GO DAG
    go_dag = load_go_dag(args.go_obo_file)
    
    # Load custom gene2go mappings
    gene2go_data = parse_custom_gene2go(args.gene2go_file)
    # Load gene symbol mappings from gene_info file
    gene_symbol_dict = load_gene_symbol_mapping(args.gene_info_file)
    
    # Save GO terms to CSV
    save_go_terms_to_csv(gene_list, gene2go_data, go_dag, gene_symbol_dict, args.output_file)
    


if __name__ == "__main__":
    main()
