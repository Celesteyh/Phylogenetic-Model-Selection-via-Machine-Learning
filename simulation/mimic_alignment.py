import os
import time
import argparse
import pandas as pd
import re


parser = argparse.ArgumentParser(description='Mimic alignments')
parser.add_argument('--iqtree_path', type=str, required=True, help='Path to iqtree')
parser.add_argument('--alignments_dir', type=str, required=True, help='Directory for the real alignments')
parser.add_argument('--parameters_path', type=str, required=True, help='Path to the parameters file')  # csv file
parser.add_argument('--trees_dir', type=str, required=True, help='Directory for the real trees')
parser.add_argument('--output_dir', type=str, required=True, help='Directory for mimicked MSAs')
args = parser.parse_args()


model_map = {'0': 'LG', '1': 'WAG', '2': 'JTT', '3': 'Q.plant', '4': 'Q.bird', '5': 'Q.mammal', '6': 'Q.pfam'}

start_time = time.time()

parameters = pd.read_csv(args.parameters_path, sep='\t')

for alignment in os.listdir(args.alignments_dir):
    # if it doesn't exist in the outpath
    if not os.path.isfile(f"{os.path.join(args.output_dir, alignment)}.log"):
        treefile = os.path.join(args.trees_dir, f"{alignment.split('.')[0]}.treefile")
        addition = alignment.split('.')[0].split('+')
        if addition[1] != '':  # if there are additional terms such as +F, +I, +G4/+Rn
            model = f"{model_map[alignment[0]]}+{'+'.join(addition[1:])}"
        else:
            model = model_map[alignment[0]]

        parameter = parameters[(parameters['ALI_ID'] == alignment.split('+')[0][1:]) & (parameters['MODEL'] == model)]
        # if there are more than one row, randomly select one
        if len(parameter) > 1:
            parameter = parameter.sample()

        if '+I' in model:
            inv_prop = parameter['PROP_INVAR'].values[0]
            model = model.replace('+I', f'+I{{{inv_prop}}}')
        if '+G4' in model:
            alpha = parameter['ALPHA'].values[0]
            model = model.replace('+G4', f'+G4{{{alpha}}}')
        elif '+R' in model:
            # convert the prop_rate to a string
            prop_rate = parameter.iloc[0, 32:48].dropna().astype(str).values.tolist()
            Rn = '+R' + re.search(r'R(\d+)', model).group(1)
            model = model.replace(Rn, f'{Rn}{{{",".join(prop_rate)}}}')

        command = f'{args.iqtree_path} --alisim {os.path.join(args.output_dir, alignment)} -s {os.path.join(args.alignments_dir, alignment)} -te {treefile} -m {model} --prefix {os.path.join(args.output_dir, alignment)} -nt 7'
        # print(command)
        os.system(command)
end_time = time.time()
print('---------------------------------------------------------------------------------')
print(f"Time taken: {end_time - start_time}")
