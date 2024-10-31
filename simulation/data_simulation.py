import argparse
import json
from scipy import stats
import os
from ete3 import Tree
import time


def sample_from_distribution(dist_name, dist_params, lbound, ubound, r):
    """
    Sample from a distribution and ensure the sample is within the bounds.
    """
    sample = round(getattr(stats, dist_name).rvs(*dist_params), r)
    while sample <= lbound or sample >= ubound:
        sample = round(getattr(stats, dist_name).rvs(*dist_params), r)
    return sample


def parse_args():
    parser = argparse.ArgumentParser(description='Data simulation')
    parser.add_argument('--iqtree_path', type=str, required=True, help='Path to iqtree')  # e.g., D:\\iqtree-2.3.4-Windows\\iqtree-2.3.4-Windows\\bin\\iqtree2
    parser.add_argument('--trees_dir', type=str, required=True, help='Directory for trees used for simulation')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for simulated MSAs')
    parser.add_argument('--distribution_file', type=str, required=True, help='Path to the fitted distribution')  # json file
    parser.add_argument('--frequencies_file', type=str, required=True, help='Path to the fitted frequencies')  # json file
    return parser.parse_args()


def generate_command_line(model_name, F, I, G4, Rn, n_taxa, length, iteration):
    # iqtree2 --alisim file_name -t tree.nwk -m model_name+F{frequencies}+I{invariant}+G4{alpha}/+Rn{w1,r1,...}
    if (model_name not in ['LG', 'WAG', 'JTT', 'Q.plant', 'Q.bird', 'Q.mammal', 'Q.pfam']) or (G4 and Rn):
        raise ValueError("Invalid model name")

    args = parse_args()
    para_dist = json.load(open(args.distribution_file))
    freq_dist = json.load(open(args.frequencies_file))

    iqtree = args.iqtree_path
    trees_dir = args.trees_dir
    output_dir = args.output_dir

    # use an arbitrary model to generate a random tree
    command_tree = f"{iqtree} --alisim gen_tree -t RANDOM{{yh/{n_taxa}}} -m LG --prefix {trees_dir}"
    os.system(f'cd /d {trees_dir} && {command_tree}')  # note this is for Windows
    # modify the branch lengths of the tree
    tree_file = trees_dir + '\\gen_tree.treefile'
    tree = open(tree_file, 'r').read()
    t = Tree(tree)
    for node in t.traverse():
        if not node.is_root():
            if node.is_leaf():
                node.dist = sample_from_distribution(para_dist['external'][0], para_dist['external'][1], 0, 10, r=10)
            else:
                node.dist = sample_from_distribution(para_dist['internal'][0], para_dist['internal'][1], 0, 10, r=10)

    file_name = f"{model_map[model_name]}t({n_taxa}){model_name}"  # v for validation; t for test
    command = f"-m {model_name}"

    if F:
        frequencies = []
        for aa in 'ARNDCQEGHILKMFPSTWYV':
            dist_name = freq_dist['FREQ_' + aa]['Best Fit Distribution']
            dist_params = freq_dist['FREQ_' + aa]['Parameters']
            frequency = sample_from_distribution(dist_name, dist_params, 0, 1, r=5)
            frequencies.append(str(frequency))
        command += '+F{' + '/'.join(frequencies) + '}'
        file_name += '+F'
    if I:
        dist_name = para_dist['invariant'][0]
        dist_params = para_dist['invariant'][1]
        invariant = sample_from_distribution(dist_name, dist_params, 0, 0.9, r=5)
        command += '+I{' + str(invariant) + '}'
        file_name += '+I{' + str(invariant) + '}'

    if G4:
        dist_name = para_dist['alpha'][0]
        dist_params = para_dist['alpha'][1]
        alpha = sample_from_distribution(dist_name, dist_params, 0.001, 11, r=5)
        command += '+G4{' + str(alpha) + '}'
        file_name += '+G4{' + str(alpha) + '}'

    elif Rn != 0:
        w_r = []
        for i in range(1, Rn + 1):
            w_name = 'R' + str(Rn) + '_w' + str(i)
            r_name = 'R' + str(Rn) + '_r' + str(i)
            w = sample_from_distribution(para_dist[w_name][0], para_dist[w_name][1], 0, 1, r=5)
            r = sample_from_distribution(para_dist[r_name][0], para_dist[r_name][1], 0.0001, 200, r=5)
            w_r.append((w, r))
        command += '+R' + str(Rn) + '{' + ','.join([str(w) + ',' + str(r) for w, r in w_r]) + '}'
        file_name += '+R' + str(Rn)

    file_name += f'[{length}]'  # only for test data
    file_name += f'_{iteration}'

    # save the new tree with the model name
    new_tree = t.write(format=1)
    new_tree_file = trees_dir + f'\\{file_name}.treefile'
    with open(new_tree_file, 'w+') as result:
        result.write(new_tree + '\n')

    command = f'{iqtree} --alisim {file_name} -t {new_tree_file} {command} --length {length}'
    os.system(f'cd /d {output_dir} && {command}')
    return


def generate_all(base_model_list, num_iterations):
    for i in range(num_iterations):
        for base_model in base_model_list:
            # for testing data
            for length in [100, 500, 1000, 2000, 3000, 4000, 5000]:
                for n_taxa in [8, 16, 32, 64, 128, 256, 512]:
                    for F in [True, False]:
                        for I in [True, False]:
                            # G4 and Rn cannot appear together
                            for G4 in [True, False]:
                                if G4:
                                    generate_command_line(base_model, F, I, True, 0, n_taxa, length, i)
                                else:
                                    for Rn in [0, 2, 3, 4]:
                                        generate_command_line(base_model, F, I, False, Rn, n_taxa, length, i)


def main():
    model_map = {'LG': 0, 'WAG': 1, 'JTT': 2, 'Q.plant': 3, 'Q.bird': 4, 'Q.mammal': 5, 'Q.pfam': 6}
    start_time = time.time()
    base_model_list = ['LG', 'WAG', 'JTT', 'Q.plant', 'Q.bird', 'Q.mammal', 'Q.pfam']
    n_replicates = 5
    generate_all(base_model_list, n_replicates)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == '__main__':
    main()
