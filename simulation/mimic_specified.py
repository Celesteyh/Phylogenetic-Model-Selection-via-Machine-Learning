import os
import sys
import pandas as pd
import time
import re

real_ali_path = 'E:/PycharmProjects/Phylogenetics/aln/real_clean_2/'
parameters_file_path = 'E:/PycharmProjects/Phylogenetics/EvoNAPS_aa_trees.txt'
treefiles = 'E:/PycharmProjects/Phylogenetics/real_treefiles/'
outpath = 'E:/PycharmProjects/Phylogenetics/mimic_specified/'

model_map = {'0': 'LG', '1': 'WAG', '2': 'JTT', '3': 'Q.plant', '4': 'Q.bird', '5': 'Q.mammal', '6': 'Q.pfam'}
reverse_model_map = {v: k for k, v in model_map.items()}

start_time = time.time()

parameters = pd.read_csv(parameters_file_path, sep='\t')

for alignment in os.listdir(real_ali_path):
    # if it doesn't exist in the outpath
    if not os.path.isfile(f"{os.path.join(outpath, alignment)}.log"):
        treefile = os.path.join(treefiles, f"{alignment.split('.')[0]}.treefile")
        # alignment = '0PF00176+F+I+G4.phy'  '0ENSG00000070831_CDC42+.phy'
        addition = alignment.split('.')[0].split('+')
        # addition = ['0ENSG00000070831_CDC42', '']  ['0PF00176', 'F', 'I', 'G4']
        if addition[1] != '':
            model = f"{model_map[alignment[0]]}+{'+'.join(addition[1:])}"
        else:
            model = model_map[alignment[0]]
        parameter = parameters[(parameters['ALI_ID'] == alignment.split('+')[0][1:]) & (parameters['MODEL'] == model)]
        # if there are more than one row, randomly select one
        if len(parameter) > 1:
            parameter = parameter.sample()
        if '+I' in model:
            inv_prop = parameter['PROP_INVAR'].values[0]
            model = model.replace('+I', f'+I{{{inv_prop}}}')  # double {
        if '+G4' in model:
            alpha = parameter['ALPHA'].values[0]
            model = model.replace('+G4', f'+G4{{{alpha}}}')
        elif '+R' in model:
            # convert the prop_rate to a string
            prop_rate = parameter.iloc[0, 32:48].dropna().astype(str).values.tolist()
            Rn = '+R' + re.search(r'R(\d+)', model).group(1)
            model = model.replace(Rn, f'{Rn}{{{",".join(prop_rate)}}}')
            # TODO: -keep-ident
        for base_model in model_map.values():
            model = model.replace(model.split('+')[0], base_model)
            # new version of iqtree
            new_alignment = reverse_model_map[model.split('+')[0]] + alignment[1:]
            command = f"E:\\iqtree-2.3.4-Windows\\iqtree-2.3.4-Windows\\bin\\iqtree2 --alisim {os.path.join(outpath, new_alignment)} -s {os.path.join(real_ali_path, alignment)} -te {treefile} -m {model} --prefix {os.path.join(outpath, new_alignment)} --num-alignments 3 -nt 7"
            # print(command)
            os.system(command)
            # remove unnecessary files if exist
            try:
                os.remove(f"{os.path.join(outpath, new_alignment)}.uniqueseq.phy")
            except FileNotFoundError:
                pass
            try:
                os.remove(f"{os.path.join(outpath, new_alignment)}.ckp.gz")
                os.remove(f"{os.path.join(outpath, new_alignment)}.iqtree")
                os.remove(f"{os.path.join(outpath, new_alignment)}.treefile")
            except Exception as e:
                print(f"{new_alignment} has an error: {e}")
                break
end_time = time.time()
print('---------------------------------------------------------------------------------')
print(f"Time taken: {end_time - start_time}")
