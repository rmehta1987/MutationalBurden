import hail as hl
import torch
import os
import numpy as np
import biomart
import mygene
import pandas as pd
from tqdm import tqdm

class MultipleOptimizer(object):
    """This is a simple class object to allow for running multiple optimizers in pytorch.

    Args:
       A list of optimizers
    """    
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def max_rows(path: str, log: bool=False):

    gene_files = os.listdir(path)
    max_row = 0
    
    for a_file in tqdm(gene_files):
        if a_file.endswith('csv'):
            bhr_phenotypes = pd.read_csv(f'{path}{a_file}',header=0,delimiter='\t')
            cur_rows = bhr_phenotypes.shape[0]
            if cur_rows > max_row:
                max_row = cur_rows
                if log:
                    print("Updated max row to: {}".format(max_row))

    
    return max_row

def add_gene_name_columns_from_mygene(dataset_to_add_to_path):
    """_summary_

    Args:
        mapped_gene_file_path (str): path to file where ensemble genes are mapped to (delimiter by ,)
        dataset_to_add_to_path (str): the dataset file to add the gene names too (delimeter by tab)

    """
    # Set up connection to server\
    df = pd.read_csv(dataset_to_add_to_path, sep='\t',header=0)
    df['Gene'] = ''
    mg = mygene.MyGeneInfo()
    for i in tqdm(range(0,df.shape[0])):
        
        ensembl_id = df['ensg'].iloc[i]
        b = mg.querymany(ensembl_id, scopes='ensembl.gene', fields='symbol', species='human') # returns a list of dicts
        one_dict = dict(*b) # unpacks dicts to one giant dict
        gene_symbol = one_dict.get('symbol', np.nan)
        if gene_symbol == np.nan:
            # try using hgnc
            ensembl_id = df['hgnc'].iloc[i]
            b = mg.querymany(ensembl_id, scopes='hgnc', fields='symbol', species='human') # returns a list of dicts
            one_dict=dict(*b)
            gene_symbol = one_dict.get('symbol', np.nan)
        if isinstance(gene_symbol,list): # get first gene name
            gene_symbol = gene_symbol[0]
        df['Gene'] = gene_symbol

    df.to_csv('s_het_estimates.genebayes_w_gene_names.tsv',sep='\t',header=True,index=False)

def add_gene_name_columns_from_a_file(mapped_gene_file_path, dataset_to_add_to_path):
    """_summary_

    Args:
        mapped_gene_file_path (str): path to file where ensemble genes are mapped to (delimiter by ,)
        dataset_to_add_to_path (str): the dataset file to add the gene names too (delimeter by tab)

    """
    mapped = pd.read_csv(mapped_gene_file_path, sep=',',header=None)
    mapped = mapped.sort_values(by=[0]).reset_index(drop=True)
    df = pd.read_csv(dataset_to_add_to_path, sep='\t',header=0)
    df = df.sort_values(by=['ensg']).reset_index(drop=True)
    df['Gene'] = ''
    df.loc[df['ensg']==mapped[0],'Gene']=mapped[1]
    df.to_csv('s_het_estimates.genebayes_w_gene_names_3.tsv',sep='\t',header=True,index=False)



#add_gene_name_columns_from_a_file('Dataset/ensn_and_gene_names.txt', '/home/rahul/PopGen/burden/Dataset/s_het_estimates.genebayes.tsv')
#add_gene_name_columns_from_mygene('/home/rahul/PopGen/burden/Dataset/s_het_estimates.genebayes.tsv')


max_row = max_rows('Bipolar/')
print("Max Number of rows was {}".format(max_row))