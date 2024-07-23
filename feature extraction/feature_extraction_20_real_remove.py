import os
import sys
import time
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

amino_acid_map = {
    '_': 0, '.': 0, '?': 0, 'A': 1, 'R': 3, 'N': 5, 'D': 7, 'C': 9, 'Q': 11, 'E': 13, 'G': 15, 'H': 17, 'I': 19,
    'L': 21, 'K': 23, 'M': 25, 'F': 27, 'P': 29, 'S': 31, 'T': 33, 'W': 35, 'Y': 37, 'V': 39
}

replacements = np.array([2, 24, 160, 896, 4608, 22528, 106496, 491520, 2228224, 9961472, 44040192, 192937984, 838860800,
                         3623878656, 15569256448, 66571993088, 283467841536, 1202590842880, 5085241278464,
                         21440476741632, 8, 32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608, 33554432,
                         134217728, 536870912, 2147483648, 8589934592, 34359738368, 137438953472, 549755813888, 6, 96,
                         384, 1536, 6144, 24576, 98304, 393216, 1572864, 6291456, 25165824, 100663296, 402653184,
                         1610612736, 6442450944, 25769803776, 103079215104, 412316860416, 1649267441664, 10, 40, 640,
                         2560, 10240, 40960, 163840, 655360, 2621440, 10485760, 41943040, 167772160, 671088640,
                         2684354560, 10737418240, 42949672960, 171798691840, 687194767360, 2748779069440, 14, 56, 224,
                         3584, 14336, 57344, 229376, 917504, 3670016, 14680064, 58720256, 234881024, 939524096,
                         3758096384, 15032385536, 60129542144, 240518168576, 962072674304, 3848290697216, 18, 72, 288,
                         1152, 18432, 73728, 294912, 1179648, 4718592, 18874368, 75497472, 301989888, 1207959552,
                         4831838208, 19327352832, 77309411328, 309237645312, 1236950581248, 4947802324992, 22, 88, 352,
                         1408, 5632, 90112, 360448, 1441792, 5767168, 23068672, 92274688, 369098752, 1476395008,
                         5905580032, 23622320128, 94489280512, 377957122048, 1511828488192, 6047313952768, 26, 104, 416,
                         1664, 6656, 26624, 425984, 1703936, 6815744, 27262976, 109051904, 436207616, 1744830464,
                         6979321856, 27917287424, 111669149696, 446676598784, 1786706395136, 7146825580544, 30, 120,
                         480, 1920, 7680, 30720, 122880, 1966080, 7864320, 31457280, 125829120, 503316480, 2013265920,
                         8053063680, 32212254720, 128849018880, 515396075520, 2061584302080, 8246337208320, 34, 136,
                         544, 2176, 8704, 34816, 139264, 557056, 8912896, 35651584, 142606336, 570425344, 2281701376,
                         9126805504, 36507222016, 146028888064, 584115552256, 2336462209024, 9345848836096, 38, 152,
                         608, 2432, 9728, 38912, 155648, 622592, 2490368, 39845888, 159383552, 637534208, 2550136832,
                         10200547328, 40802189312, 163208757248, 652835028992, 2611340115968, 10445360463872, 42, 168,
                         672, 2688, 10752, 43008, 172032, 688128, 2752512, 11010048, 176160768, 704643072, 2818572288,
                         11274289152, 45097156608, 180388626432, 721554505728, 2886218022912, 11544872091648, 46, 184,
                         736, 2944, 11776, 47104, 188416, 753664, 3014656, 12058624, 48234496, 771751936, 3087007744,
                         12348030976, 49392123904, 197568495616, 790273982464, 3161095929856, 12644383719424, 50, 200,
                         800, 3200, 12800, 51200, 204800, 819200, 3276800, 13107200, 52428800, 209715200, 3355443200,
                         13421772800, 53687091200, 214748364800, 858993459200, 3435973836800, 13743895347200, 54, 216,
                         864, 3456, 13824, 55296, 221184, 884736, 3538944, 14155776, 56623104, 226492416, 905969664,
                         14495514624, 57982058496, 231928233984, 927712935936, 3710851743744, 14843406974976, 58, 232,
                         928, 3712, 14848, 59392, 237568, 950272, 3801088, 15204352, 60817408, 243269632, 973078528,
                         3892314112, 62277025792, 249108103168, 996432412672, 3985729650688, 15942918602752, 62, 248,
                         992, 3968, 15872, 63488, 253952, 1015808, 4063232, 16252928, 65011712, 260046848, 1040187392,
                         4160749568, 16642998272, 266287972352, 1065151889408, 4260607557632, 17042430230528, 66, 264,
                         1056, 4224, 16896, 67584, 270336, 1081344, 4325376, 17301504, 69206016, 276824064, 1107296256,
                         4429185024, 17716740096, 70866960384, 1133871366144, 4535485464576, 18141941858304, 70, 280,
                         1120, 4480, 17920, 71680, 286720, 1146880, 4587520, 18350080, 73400320, 293601280, 1174405120,
                         4697620480, 18790481920, 75161927680, 300647710720, 4810363371520, 19241453486080, 74, 296,
                         1184, 4736, 18944, 75776, 303104, 1212416, 4849664, 19398656, 77594624, 310378496, 1241513984,
                         4966055936, 19864223744, 79456894976, 317827579904, 1271310319616, 20340965113856, 78, 312,
                         1248, 4992, 19968, 79872, 319488, 1277952, 5111808, 20447232, 81788928, 327155712, 1308622848,
                         5234491392, 20937965568, 83751862272, 335007449088, 1340029796352, 5360119185408],
                        dtype=np.uint64)


def calculate_base_freq(msa, gaps):
    """
    Calculate the base frequencies of every sequence in the MSA
    """
    n_sites, n_taxa = msa.shape[0], msa.shape[1]
    # TODO: change the precision of base_freqs???
    base_freqs = np.zeros((n_taxa, 20), dtype=np.float16)  # (n_taxa, n_aa) increase the precision here!!!!

    for i_taxa in range(n_taxa):
        seq = msa[:, i_taxa]
        if gaps:
            seq = seq[seq != 0]
        seq_adjusted = (seq - 1) // 2
        counts = np.bincount(seq_adjusted, minlength=20)
        # print(counts)
        base_freqs[i_taxa] = counts / float(np.sum(counts))  # changed n_sites to np.sum(counts)
    # for i in range(n_taxa):
    #     if np.sum(base_freqs[i]) != 1:
    #         print(base_freqs[i])
    #         print(np.sum(base_freqs[i]))
    return base_freqs


# def calculate_aa_entropy(msa):
#     unique_aa, counts = np.unique(msa, return_counts=True)
#     probs_aa = counts / counts.sum()
#     aa_entropy = -np.sum(probs_aa * np.log2(probs_aa))
#     return aa_entropy


def remove_gaps(seq_1, seq_2):
    valid_sites = (seq_1 != 0) & (seq_2 != 0)
    return seq_1[valid_sites], seq_2[valid_sites]


# TODO: change the precision of the features??????????????????????????????????????
def make_profile(m, num_sumstats=1225):
    n_taxa = m.shape[0]
    n_sites = m.shape[1]
    # mm is the extracted features of the MSA, which contains num_sumstats pairs and 440 features for each pair
    # invariant # 20, replacements between groups # 380, base frequencies # 40,  aa shannon entropy # 1
    mm = np.zeros((num_sumstats, 440), dtype=np.float16)
    m = m.T  # (n_sites, n_taxa)
    d = np.zeros(m.shape, dtype=np.uint16)
    for k, v in amino_acid_map.items():
        d[m == k] = v
    m = d
    # m = np.vectorize(amino_acid_map.get)(m)
    # get the base frequencies of each sequence
    base_freqs = calculate_base_freq(m, gaps=True)
    # get the shannon entropy for amino acids
    # aa_entropy = calculate_aa_entropy(m)

    # np.uint16 makes the maximum number of taxa 65535
    pairs = np.random.choice(n_taxa, (num_sumstats, 2)).astype(np.uint16)
    for i in range(num_sumstats):
        if pairs[i, 0] == pairs[i, 1]:
            pairs[i] = np.random.choice(n_taxa, 2, replace=False)

    unique_pairs, indices = np.unique(pairs, axis=0, return_inverse=True)  # unique pairs are sorted (directional)

    # calculate replacement frequencies for each unique pair
    unique_replacement_freqs = np.zeros((unique_pairs.shape[0], 400), dtype=np.float16)
    for i_pair, pair in enumerate(unique_pairs):
        replacement_count = np.zeros(400, dtype=np.float16)

        pair = remove_gaps(m[:, pair[0]], m[:, pair[1]])

        # This type conversion is necessary
        # replacement = m[:, pair[0]].astype(np.uint64) << m[:, pair[1]].astype(np.uint64)
        replacement = pair[0].astype(np.uint64) << pair[1].astype(np.uint64)
        if len(replacement) == 0:
            replacement_count = np.zeros(400, dtype=np.float16)  # no sites left
        else:
            for i, r in enumerate(replacements):
                count = np.sum(replacement == r)
                replacement_count[i] = count / float(len(replacement))
        unique_replacement_freqs[i_pair] = replacement_count

    for i_pair in range(0, num_sumstats):
        first_seq, second_seq = pairs[i_pair]
        # base frequencies for the first and second sequences
        mm[i_pair, 400:420] = base_freqs[first_seq]
        mm[i_pair, 420:] = base_freqs[second_seq]

        mm[i_pair, :400] = unique_replacement_freqs[indices[i_pair]]
        # mm[:, 440] = aa_entropy
    return mm


def sitewise_aa_freq(m):
    n_taxa = m.shape[0]
    n_sites = m.shape[1]
    mm = np.zeros((n_sites, 20), dtype=np.float32)
    for i in range(1, 40, 2):  # note the range here
        mm[:, i] = np.sum(m == i, axis=0) / n_taxa  # (1, n_sites) normalizing by n_taxa so that the sum of each row is 1
    return mm  # (n_sites, 20)


def make_RHAS(m, num_sumstats=10000):
    # gaps are naturally ignored in this function
    n_taxa = m.shape[0]
    n_sites = m.shape[1]

    # msa = np.array([list(y) for y in m])

    d = np.zeros(m.shape, dtype=np.uint16)
    for k, v in amino_acid_map.items():
        d[m == k] = v
    m = d

    # TODO: test this condition
    if num_sumstats > n_sites:
        # pre-calculate the frequencies of each site
        m = sitewise_aa_freq(m)  # (n_sites, 20), this is a matrix of frequencies of each site
        idx = np.random.randint(0, n_sites, size=num_sumstats, dtype=int)  # sample num_sumstats sites
        m = m[idx, :]  # (num_sumstats, 20)
    elif num_sumstats == n_sites:
        m = sitewise_aa_freq(m)
    else:
        idx = np.random.randint(0, n_sites, size=num_sumstats, dtype=int)
        m = m[:, idx]  # (n_taxa, num_sumstats)
        m = sitewise_aa_freq(m)  # (num_sumstats, 20)

    return m  # (num_sumstats, 20)


# def extract(file):
#     with open(path1 + file, 'rb') as f:
#         lines = f.readlines()
#         # texts = [x.decode().strip().split(' ')[-1].split('\t')[1] for x in lines[1:]]
#         texts = [x.decode().strip().split('\t')[1].upper() for x in lines[1:]]
#         raw_msa = np.array(texts)
#         # raw_msa = np.array([x.decode().strip().split(' ')[-1] for x in lines[1:]])
#         msa = np.array([list(s) for s in raw_msa])  # (n_taxa, n_sites)
#         feature = make_profile(msa)
#         # save the feature to a file in path2
#         np.save(path2 + file[:-4] + '.npy', feature)
#         # delete the original file
#         os.remove(path1 + file)


if __name__ == '__main__':
    # start_time = time.time()
    # path1 = '/scratch/dx61/yd7308/validation_MSAs/'
    # # path2 = '/scratch/dx61/yd7308/20_validation_feature/'
    # path2 = '/scratch/dx61/yd7308/real_feature_remove/'
    # files = os.listdir(path1)
    # with ProcessPoolExecutor(max_workers=288) as executor:
    #     executor.map(extract, files)
    # path1 = 'E:/PycharmProjects/Phylogenetics/simulation_test/0t(8)LG+F+I{0.07806}[100]_0.phy'
    # path2 = 'E:/PycharmProjects/Phylogenetics/simulation_test/0t(8)LG+F+I{0.53524}+R2[100]_0.phy'
    # path8 = 'E:/PycharmProjects/Phylogenetics/simulation_test/6t(8)Q.pfam+I{0.40589}+G4{0.89189}[1000]_1.phy'
    # path16 = 'E:/PycharmProjects/Phylogenetics/simulation_test/6t(16)Q.pfam+I{0.18396}+G4{0.62005}[1000]_1.phy'
    # path32 = 'E:/PycharmProjects/Phylogenetics/simulation_test/6t(32)Q.pfam+F+I{0.27834}+G4{3.00406}[1000]_1.phy'
    # path64 = 'E:/PycharmProjects/Phylogenetics/simulation_test/6t(64)Q.pfam+F+I{0.26955}+R3[1000]_1.phy'
    # path128 = 'E:/PycharmProjects/Phylogenetics/simulation_test/6t(128)Q.pfam+I{0.35137}+G4{1.31004}[1000]_1.phy'
    real1 = "E:\PycharmProjects\Phylogenetics\mimicked_validation_alignments/0ENSG00000132639_SNAP25+.phy_16.phy"
    real2 = "E:/PycharmProjects/Phylogenetics/aln/real/6PF01459+F+G4.phy"
    reals = [real1]
    p = 'E:\\PycharmProjects\\Phylogenetics\\b_mimicked_training'
    # p = 'E:/3_types_of_ending/r'
    # # files = [path1, path2, path8, path16, path32, path64, path128]
    # # files = [path128]
    for ali in os.listdir(p)[:1]:
    # for path in reals:
        # for ali in os.listdir(p):
        start_time = time.time()
        # with open(path, 'rb') as f:
        with open(os.path.join(p, ali), 'rb') as f:
            lines = f.readlines()
            print(lines[0])
            # this solves the problem of no ' ' between sequence name and sequence by finding the number of sites
            # and selecting the last n_sites characters
            # num_sites = int(lines[0].split(b'\r')[0].split(b' ')[1].decode())
            # # this if statement solves n_taxa n_sites\r\n => ignore this line
            # # print(lines[0])
            # if len(lines[0].decode().strip()[-num_sites:]) != num_sites:
            #     # lines = lines[1:]
            #     print(lines[0])
            #     print(ali)
            # texts = [x.decode().strip()[-num_sites:].upper() for x in lines]
            # # texts = [x.decode().strip().split('\t')[1].upper() for x in lines[1:]]
            # # raw_msa = np.array([x.strip().split(' ')[-1] for x in lines[1:]])
            # raw_msa = np.array(texts)
            # print(raw_msa)
            # msa = np.array([list(s) for s in raw_msa])  # (n_taxa, n_sites)
            # # print(msa[0])
            # feature = make_profile(msa)
            # # print('----------------------------')
            # # print(feature[0, 0:20])
            # # print(feature[1, 0:20])
            # # print(feature[2, 0:20])
            # # print(feature[0, 20:380])
            # # print(feature[1, 20:380])
            # print(np.sum(feature[0, 400:420]))
            # print(np.sum(feature[0, 420:]))
            # # print(feature[0, 400: 440])
            # # print(feature[1, 400: 440])
            # # print(feature[2, 400: 440])
            # end_time = time.time()
            # print(f"Time taken: {end_time - start_time} seconds")
