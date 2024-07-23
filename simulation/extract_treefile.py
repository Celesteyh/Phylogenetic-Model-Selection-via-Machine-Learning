import os
import sys

import pandas as pd
import random

csv_path = 'E:/PycharmProjects/Phylogenetics/EvoNAPS_aa_trees.txt'
ali_path = 'E:/PycharmProjects/Phylogenetics/aln/real'
outpath = 'E:/PycharmProjects/Phylogenetics/real_treefiles'

df = pd.read_csv(csv_path, on_bad_lines='warn', sep='\t')
df = df[['ALI_ID', 'NEWICK_STRING']]

for alignment in os.listdir(ali_path):
    ali_name = alignment[1:].split('+')[0]
    trees = df[df['ALI_ID'] == ali_name]['NEWICK_STRING'].values
    if len(trees) == 0:
        print(f"No tree found for {alignment}")
        sys.exit(0)
    # randomly select a tree if there are multiple trees
    if len(trees) > 1:
        tree = random.choice(trees)
    else:
        tree = trees[0]
    # save the tree to a new file
    with open(os.path.join(outpath, f"{alignment.split('.')[0]}.treefile"), 'w') as file:
        file.write(tree)
