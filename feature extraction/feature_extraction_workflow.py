import argparse
import os
import time
import numpy as np
from scipy.stats import entropy
from concurrent.futures import ProcessPoolExecutor
import pickle

amino_acid_map = {
    'A': 1, 'R': 3, 'N': 5, 'D': 7, 'C': 9, 'Q': 11, 'E': 13, 'G': 15, 'H': 17, 'I': 19,
    'L': 21, 'K': 23, 'M': 25, 'F': 27, 'P': 29, 'S': 31, 'T': 33, 'W': 35, 'Y': 37, 'V': 39
}

# the idea of using the left-shift operator was inspired by ModelRevelator
replacements = np.array(
    [1 << 1, 3 << 3, 5 << 5, 7 << 7, 9 << 9, 11 << 11, 13 << 13, 15 << 15, 17 << 17, 19 << 19, 21 << 21, 23 << 23,
     25 << 25, 27 << 27, 29 << 29, 31 << 31, 33 << 33, 35 << 35, 37 << 37, 39 << 39,
     1 << 3, 1 << 5, 1 << 7, 1 << 9, 1 << 11, 1 << 13, 1 << 15, 1 << 17, 1 << 19, 1 << 21, 1 << 23, 1 << 25, 1 << 27,
     1 << 29, 1 << 31, 1 << 33, 1 << 35, 1 << 37, 1 << 39,
     3 << 1, 3 << 5, 3 << 7, 3 << 9, 3 << 11, 3 << 13, 3 << 15, 3 << 17, 3 << 19, 3 << 21, 3 << 23, 3 << 25, 3 << 27,
     3 << 29, 3 << 31, 3 << 33, 3 << 35, 3 << 37, 3 << 39,
     5 << 1, 5 << 3, 5 << 7, 5 << 9, 5 << 11, 5 << 13, 5 << 15, 5 << 17, 5 << 19, 5 << 21, 5 << 23, 5 << 25, 5 << 27,
     5 << 29, 5 << 31, 5 << 33, 5 << 35, 5 << 37, 5 << 39,
     7 << 1, 7 << 3, 7 << 5, 7 << 9, 7 << 11, 7 << 13, 7 << 15, 7 << 17, 7 << 19, 7 << 21, 7 << 23, 7 << 25, 7 << 27,
     7 << 29, 7 << 31, 7 << 33, 7 << 35, 7 << 37, 7 << 39,
     9 << 1, 9 << 3, 9 << 5, 9 << 7, 9 << 11, 9 << 13, 9 << 15, 9 << 17, 9 << 19, 9 << 21, 9 << 23, 9 << 25, 9 << 27,
     9 << 29, 9 << 31, 9 << 33, 9 << 35, 9 << 37, 9 << 39,
     11 << 1, 11 << 3, 11 << 5, 11 << 7, 11 << 9, 11 << 13, 11 << 15, 11 << 17, 11 << 19, 11 << 21, 11 << 23, 11 << 25,
     11 << 27, 11 << 29, 11 << 31, 11 << 33, 11 << 35, 11 << 37, 11 << 39,
     13 << 1, 13 << 3, 13 << 5, 13 << 7, 13 << 9, 13 << 11, 13 << 15, 13 << 17, 13 << 19, 13 << 21, 13 << 23, 13 << 25,
     13 << 27, 13 << 29, 13 << 31, 13 << 33, 13 << 35, 13 << 37, 13 << 39,
     15 << 1, 15 << 3, 15 << 5, 15 << 7, 15 << 9, 15 << 11, 15 << 13, 15 << 17, 15 << 19, 15 << 21, 15 << 23, 15 << 25,
     15 << 27, 15 << 29, 15 << 31, 15 << 33, 15 << 35, 15 << 37, 15 << 39,
     17 << 1, 17 << 3, 17 << 5, 17 << 7, 17 << 9, 17 << 11, 17 << 13, 17 << 15, 17 << 19, 17 << 21, 17 << 23, 17 << 25,
     17 << 27, 17 << 29, 17 << 31, 17 << 33, 17 << 35, 17 << 37, 17 << 39,
     19 << 1, 19 << 3, 19 << 5, 19 << 7, 19 << 9, 19 << 11, 19 << 13, 19 << 15, 19 << 17, 19 << 21, 19 << 23, 19 << 25,
     19 << 27, 19 << 29, 19 << 31, 19 << 33, 19 << 35, 19 << 37, 19 << 39,
     21 << 1, 21 << 3, 21 << 5, 21 << 7, 21 << 9, 21 << 11, 21 << 13, 21 << 15, 21 << 17, 21 << 19, 21 << 23, 21 << 25,
     21 << 27, 21 << 29, 21 << 31, 21 << 33, 21 << 35, 21 << 37, 21 << 39,
     23 << 1, 23 << 3, 23 << 5, 23 << 7, 23 << 9, 23 << 11, 23 << 13, 23 << 15, 23 << 17, 23 << 19, 23 << 21, 23 << 25,
     23 << 27, 23 << 29, 23 << 31, 23 << 33, 23 << 35, 23 << 37, 23 << 39,
     25 << 1, 25 << 3, 25 << 5, 25 << 7, 25 << 9, 25 << 11, 25 << 13, 25 << 15, 25 << 17, 25 << 19, 25 << 21, 25 << 23,
     25 << 27, 25 << 29, 25 << 31, 25 << 33, 25 << 35, 25 << 37, 25 << 39,
     27 << 1, 27 << 3, 27 << 5, 27 << 7, 27 << 9, 27 << 11, 27 << 13, 27 << 15, 27 << 17, 27 << 19, 27 << 21, 27 << 23,
     27 << 25, 27 << 29, 27 << 31, 27 << 33, 27 << 35, 27 << 37, 27 << 39,
     29 << 1, 29 << 3, 29 << 5, 29 << 7, 29 << 9, 29 << 11, 29 << 13, 29 << 15, 29 << 17, 29 << 19, 29 << 21, 29 << 23,
     29 << 25, 29 << 27, 29 << 31, 29 << 33, 29 << 35, 29 << 37, 29 << 39,
     31 << 1, 31 << 3, 31 << 5, 31 << 7, 31 << 9, 31 << 11, 31 << 13, 31 << 15, 31 << 17, 31 << 19, 31 << 21, 31 << 23,
     31 << 25, 31 << 27, 31 << 29, 31 << 33, 31 << 35, 31 << 37, 31 << 39,
     33 << 1, 33 << 3, 33 << 5, 33 << 7, 33 << 9, 33 << 11, 33 << 13, 33 << 15, 33 << 17, 33 << 19, 33 << 21, 33 << 23,
     33 << 25, 33 << 27, 33 << 29, 33 << 31, 33 << 35, 33 << 37, 33 << 39,
     35 << 1, 35 << 3, 35 << 5, 35 << 7, 35 << 9, 35 << 11, 35 << 13, 35 << 15, 35 << 17, 35 << 19, 35 << 21, 35 << 23,
     35 << 25, 35 << 27, 35 << 29, 35 << 31, 35 << 33, 35 << 37, 35 << 39,
     37 << 1, 37 << 3, 37 << 5, 37 << 7, 37 << 9, 37 << 11, 37 << 13, 37 << 15, 37 << 17, 37 << 19, 37 << 21, 37 << 23,
     37 << 25, 37 << 27, 37 << 29, 37 << 31, 37 << 33, 37 << 35, 37 << 39,
     39 << 1, 39 << 3, 39 << 5, 39 << 7, 39 << 9, 39 << 11, 39 << 13, 39 << 15, 39 << 17, 39 << 19, 39 << 21, 39 << 23,
     39 << 25, 39 << 27, 39 << 29, 39 << 31, 39 << 33, 39 << 35, 39 << 37], dtype=np.uint64)

# amino acid frequencies for each substitution model
predefined_freqs = {'0': [0.07906592, 0.05594094, 0.04197696, 0.05305195, 0.01293699,
                          0.04076696, 0.07158593, 0.05733694, 0.02235498, 0.06215694,
                          0.0990809, 0.06459994, 0.02295098, 0.04230196, 0.04403996,
                          0.06119694, 0.05328695, 0.01206599, 0.03415497, 0.06914693],
                    '1': [0.08662791, 0.043972, 0.0390894, 0.05704511, 0.0193078, 0.0367281,
                          0.05805891, 0.08325181, 0.0244313, 0.048466, 0.08620901,
                          0.06202861, 0.0195027, 0.0384319, 0.0457631, 0.06951791,
                          0.06101271, 0.0143859, 0.0352742, 0.07089561],
                    '2': [0.07674792, 0.05169095, 0.04264496, 0.05154395, 0.01980298,
                          0.04075196, 0.06182994, 0.07315193, 0.02294398, 0.05376095,
                          0.09190391, 0.05867594, 0.02382598, 0.04012596, 0.05090095,
                          0.06876493, 0.05856494, 0.01426099, 0.03210197, 0.06600493],
                    '3': [0.074923, 0.0505, 0.038734, 0.053195, 0.0113, 0.037499, 0.068513,
                          0.059627, 0.021204, 0.058991, 0.102504, 0.067306, 0.022371,
                          0.043798, 0.037039, 0.084451, 0.04785, 0.012322, 0.030777,
                          0.077097],
                    '4': [0.066363, 0.054021, 0.037784, 0.047511, 0.022651, 0.048841,
                          0.071571, 0.058368, 0.025403, 0.045108, 0.100181, 0.061361,
                          0.021069, 0.03823, 0.053861, 0.089298, 0.053536, 0.012313,
                          0.027173, 0.065359],
                    '5': [0.067997, 0.055503, 0.036288, 0.046867, 0.021435, 0.050281,
                          0.068935, 0.055323, 0.02641, 0.041953, 0.101191, 0.060037,
                          0.019662, 0.036237, 0.055146, 0.096864, 0.057136, 0.011785,
                          0.02473, 0.066223],
                    '6': [0.085788, 0.057731, 0.042028, 0.056462, 0.010447, 0.039548,
                          0.067799, 0.064861, 0.02104, 0.055398, 0.100413, 0.059401,
                          0.019898, 0.042789, 0.039579, 0.069262, 0.055498, 0.01443,
                          0.033233, 0.064396],
                    }


def parse_args():
    parser = argparse.ArgumentParser(description='Feature extraction')
    parser.add_argument('--MSA_path', type=str, required=True, help='MSAs')
    parser.add_argument('--feature1_path', type=str, required=True, help='Feature for protFinder')
    parser.add_argument('--feature2_path', type=str, required=True, help='Feature for RHASFinder')
    parser.add_argument('--feature3_path', type=str, required=True, help='Feature for protFFinder')
    parser.add_argument('--time_cost_path', type=str, help='Time cost list')
    return parser.parse_args()


args = parse_args()
path1 = args.MSA_path
path2 = args.feature1_path
path3 = args.feature2_path
path4 = args.feature3_path


def convert_msa(file):
    """
    Convert the .phy file to an array
    Note that 'temp' are different for different file sources:
    Simulated: temp = np.array([x.decode().strip().split(' ')[-1].upper() for x in lines[1:]])
    Mimicked: temp = np.array([x.decode().strip().split(' ')[-1].upper() for x in lines])
    Real: temp = np.array([x.decode().strip().split('\t')[-1].upper() for x in lines[1:]])
    """
    with open(path1 + file, 'rb') as f:
        lines = f.readlines()
        temp = np.array([x.decode().strip().split(' ')[-1].upper() for x in lines[1:]])
        msa = np.array([list(s) for s in temp])  # (n_taxa, n_sites)
    return msa


def calculate_seq_freq(msa, n_taxa, n_sites, new_lengths=None):
    """
    Calculate the amino acid frequencies of each sequence
    """
    seq_freqs = np.zeros((n_taxa, 20), dtype=np.float16)
    for i in range(20):
        count = np.sum(msa == i * 2 + 1, axis=1)
        seq_freqs[:, i] = count / n_sites if new_lengths is None else count / new_lengths
    return seq_freqs


def calculate_site_freq(msa, n_taxa, n_sites, new_lengths=None):
    """
    Calculate the amino acid frequencies of each site
    """
    site_freqs = np.zeros((n_sites, 20), dtype=np.float32)
    for i in range(20):
        count = np.sum(msa == i * 2 + 1, axis=0)
        site_freqs[:, i] = count / n_taxa if new_lengths is None else count / new_lengths
    return site_freqs


def remove_gaps(seq_1, seq_2):
    """
    The site is removed if any of the sequences has a gap at that site
    """
    valid_sites = (seq_1 != 0) & (seq_2 != 0)
    return seq_1[valid_sites], seq_2[valid_sites]


def replace_gaps(site):
    """
    Replace gaps with the most common amino acid in the site
    """
    # remove gaps
    temp = site[site != 0]
    if len(temp) == 0:
        # randomly choose an amino acid if all gaps
        site[site == 0] = np.random.choice(20) * 2 + 1
    else:
        temp = (temp - 1) // 2
        counts = np.bincount(temp, minlength=20)
        # find the most common amino acids
        most_common_amino_acids = np.flatnonzero(counts == counts.max())
        # randomly select one if there is a tie
        most_common = np.random.choice(most_common_amino_acids)
        site[site == 0] = most_common * 2 + 1
    return site


def feature_extraction(file, num_pairs=625, num_sites=2000, gap=None):
    """
    File conversion, feature extraction, and time cost calculation
    """
    if gap not in ['ignore', 'replace', None]:
        raise ValueError("gap must be one of 'ignore', 'replace', or None")

    start = time.time()
    m = convert_msa(file)

    time_point_1 = time.time()
    # ---------------------------------------------protFinder-------------------------------------------------
    n_taxa = m.shape[0]
    n_sites = m.shape[1]
    feature = np.zeros([num_pairs, 440], dtype=np.float16)

    d = np.zeros(m.shape, dtype=np.uint16)  # any symbol other than the 20 amino acids is 0
    for k, v in amino_acid_map.items():
        d[m == k] = v
    m = d

    if gap == 'replace':
        m = np.apply_along_axis(replace_gaps, 0, m)
    if gap == 'ignore':
        # obtain the length of each sequence without gaps
        new_length_1 = np.sum(m != 0, axis=1)
        seq_freqs = calculate_seq_freq(m, n_taxa, n_sites, new_length_1)
    else:
        # no gaps or replace
        seq_freqs = calculate_seq_freq(m, n_taxa, n_sites)

    # np.uint16 makes the maximum number of taxa 65535
    pairs = np.random.choice(n_taxa, (num_pairs, 2)).astype(np.uint16)
    for i in range(num_pairs):
        if pairs[i, 0] == pairs[i, 1]:
            pairs[i] = np.random.choice(n_taxa, 2, replace=False)

    unique_pairs, indices = np.unique(pairs, axis=0, return_inverse=True)  # unique pairs are sorted (directional)

    # calculate replacement frequencies for each unique pair
    unique_replacement_freqs = np.zeros((unique_pairs.shape[0], 400), dtype=np.float16)
    for i_pair, pair in enumerate(unique_pairs):
        replacement_count = np.zeros(400, dtype=np.float16)
        if gap == 'ignore':
            pair = remove_gaps(m[pair[0], :], m[pair[1], :])
            # this type conversion is necessary
            replacement = pair[0].astype(np.uint64) << pair[1].astype(np.uint64)
        else:
            replacement = m[pair[0], :].astype(np.uint64) << m[pair[1], :].astype(np.uint64)

        for i, r in enumerate(replacements):
            count = np.sum(replacement == r)
            replacement_count[i] = count / float(len(replacement))
        unique_replacement_freqs[i_pair] = replacement_count

    for i_pair in range(0, num_pairs):
        first_seq, second_seq = pairs[i_pair]
        feature[i_pair, 400:420] = seq_freqs[first_seq]
        feature[i_pair, 420:] = seq_freqs[second_seq]
        feature[i_pair, :400] = unique_replacement_freqs[indices[i_pair]]

    time_point_2 = time.time()
    # ---------------------------------------------RHASFinder-------------------------------------------------
    # # randomly sample 2000 sites with repetitions
    # sites = np.random.choice(n_sites, num_sites, replace=True)
    # selected = m[:, sites]
    # # selected = m[sites, :]
    #
    # site_freqs = calculate_site_freq(selected, n_taxa, n_sites)  # (n_sites, 20), aa frequencies of each site
    # # calculate entropy for each site
    # entropies = np.sort(entropy(site_freqs, base=2, axis=1))
    if gap == 'ignore':
        # obtain the length of each site without gaps
        new_lengths = np.sum(m != 0, axis=0)
        all_site_freqs = calculate_site_freq(m, n_taxa, n_sites, new_lengths)
    else:
        all_site_freqs = calculate_site_freq(m, n_taxa, n_sites)

    sites = np.random.choice(n_sites, num_sites, replace=True)
    selected = all_site_freqs[sites, :]
    entropies = np.sort(entropy(selected, base=2, axis=1))

    time_point_3 = time.time()
    # ---------------------------------------------protFFinder------------------------------------------------
    # calculate overall 20 aa frequencies
    # note that here overall_freq cannot be np.float16 otherwise np.sum(overall_freq) will overflow
    overall_freq = np.zeros(20, dtype=np.float32)
    for i in range(20):
        overall_freq[i] = np.sum(m == i * 2 + 1)
    # normalize
    overall_freq = overall_freq / np.sum(overall_freq)
    # calculate the kl divergence between overall_freq and all predefined_freqs
    kl_divergences = np.zeros(7, dtype=np.float32)
    for i in range(7):
        kl_divergences[i] = entropy(overall_freq, predefined_freqs[str(i)])

    time_point_4 = time.time()

    time_convert = time_point_1 - start
    time_feature = time_point_2 - time_point_1
    time_RHAS = time_point_3 - time_point_2
    time_F = time_point_4 - time_point_3
    time_costs = {
        "file": file,
        "time_convert": time_convert,
        "time_feature": time_feature,
        "time_RHAS": time_RHAS,
        "time_F": time_F
    }

    return feature.reshape((25, 25, 440)), entropies.reshape((40, 50)), kl_divergences, time_costs


def extract(file):
    try:
        feature_1, feature_2, feature_3, time = feature_extraction(file)
        np.save(path2 + file[:-4] + '.npy', feature_1)
        np.save(path3 + file[:-4] + '.npy', feature_2)
        np.save(path4 + file[:-4] + '.npy', feature_3)
        return time
    except Exception as e:
        print(f"{file} has an error: {e}")
        return


def main():
    # time_cost_list = []
    start_time = time.time()
    files = os.listdir(path1)
    with ProcessPoolExecutor(max_workers=40) as executor:
        executor.map(extract, files)
        # for result in executor.map(extract, files):
        #     if result is not None:
        #         time_cost_list.append(result)
    # # save the time costs
    # with open(time_cost_path, 'wb') as f:
    #     pickle.dump(time_cost_list, f)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == '__main__':
    main()
