#%%
import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

#folder = ""
### These two never called? 
def load_drug_list(config):
    filename = config['druglist_file']
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid(config):
    drugs = load_drug_list(config)
    drug_id = []
    datas = []
    outputfile = open(config['data_dir'] + '/pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(config['data_dir']+ "/unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source(config):
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(config['datadir'] + "/small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(config['datadir'] + "/unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict(config):
    reader = csv.reader(open(config['datadir'] + "/pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles(config):
    cids_dict = load_cid_dict(config)
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', config['datadir'] + '/drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(config['datadir'] + '/drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(config['datadir'] + "/drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def load_drug_smile(config):
    reader = csv.reader(open(config['drug_smiles_file']))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[2]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    
    return drug_dict, drug_smile, smile_graph

def save_cell_mut_matrix(config):
    
    genelist = pd.read_csv(config['genelist_file'],names=['gene_id']).iloc[:,0].to_numpy()
    mutation_data = pd.read_csv(config['genetic_feature_file']).pipe(lambda x: x[x.gene_id.isin(genelist)])

    ### Need to change this to handle not 
    ### having 0s in genetic feature file 
    ### as well as allow for an input genelist
    ### On a second pass this code seems like it willl be better if i convert to pandas 
    res = pd.pivot(
        mutation_data[["pgxpdc_id", "gene_id", "is_mutated"]], 
        index = "pgxpdc_id",
        columns = "gene_id",
        values="is_mutated"
    ).fillna(0)

    mut_dict = {gid:idx for idx,gid in enumerate(res.columns.tolist()) }
    cell_dict = {cid:idx for idx,cid in enumerate(res.index.tolist())}
    feature_matrix = res.to_numpy()

    with open(config['outdir']+ '/mut_dict.pickle', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    
    return cell_dict,feature_matrix


"""
Reworked this, so that it saves the data given a single matrix; this way 
we can separately prep the 
"""
def save_drug_cell_matrix(config):
    f = open(config['drug_auc_file'])
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix(config)
    drug_dict, drug_smile, smile_graph = load_drug_smile(config)

    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    for item in reader:
        drug = item[4]
        cell = item[0]
        auc = float(item[5])#######********CHANGE THIS**************
        
        temp_data.append((drug, cell, auc))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []
    random.shuffle(temp_data)
    ## refactor to not use auc 

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            ## I feel like this is wrong, but it seems to work somehow
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict[cell]] = 1
            lst_drug.append(drug)
            lst_cell.append(cell)
        
    with open(f"{config['outdir']}/drug_dict.pickle", 'wb+') as fp:
        pickle.dump(drug_dict, fp)

    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    data =  TestbedDataset(root=config['outdir'], dataset=config['dataset_name'], xd=xd, xt=xc, y=y, smile_graph=smile_graph)
    return data 

#%%
with open("ctrpv2ccle_config.json") as infl:
    config = json.load(infl)
ds = save_drug_cell_matrix(config)

#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--config', type=str, required=False, default=0, help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
    args = parser.parse_args()
    with open(args.config) as infl:
        config = json.load(infl)
    save_drug_cell_matrix(config)
