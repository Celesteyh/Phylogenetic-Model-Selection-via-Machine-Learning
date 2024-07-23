import sys
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import argparse
import os
import time

DATA_DIR = "E:/PycharmProjects/Phylogenetics"
ALN_DIR = "/aln"
FORWARD_SLASH = "/"
ALN_FORMAT = ".phy"
ALN_INFO_TABLE = "aa_alignments_selected.csv"
SEQ_TABLE = "aa_sequences.csv"


def reconstruct_alignment(data):
    ali_id, n_taxa, n_sites = data[1], int(float(data[2])), int(float(data[3]))
    # set the output file path
    output_ali = DATA_DIR + ALN_DIR + FORWARD_SLASH + ali_id + ALN_FORMAT
    # recover the aln (if not existed)
    if not (os.path.isfile(output_ali) and os.path.getsize(output_ali) > 0):
        # create a list of size n_taxa
        sequences = [0 for _ in range(n_taxa)]
        all_sequences.seek(0)  # Reset file pointer to the beginning of the file
        next(all_sequences)
        for line in all_sequences:
            info = line.split(',')  # info[1]: ALI_ID, info[2]: SEQ_INDEX, info[3]: SEQ_NAME, info[-1]: SEQ
            if info[1] == ali_id:
                sequences[int(info[2]) - 1] = [info[3], info[-1]]

        result = open(output_ali, 'w')

        # write the first line
        result.write('{0}\t{1}\n'.format(str(n_taxa), str(n_sites)))

        # write all sequences
        for seq in sequences:
            result.write('{0}\t{1}'.format(seq[0], seq[1]))

        result.close()


if __name__ == '__main__':
    start_time = time.time()
    # open the alignment info table
    # every line in the table is the information of an alignment including ALI_KEY, ALI_ID, TAXA, SITES
    alignments = open(os.path.join(DATA_DIR, ALN_INFO_TABLE), 'r')
    next(alignments)    # open the sequences table, \n is very important!!!
    all_sequences = open(os.path.join(DATA_DIR, SEQ_TABLE), 'r', newline='\n')
    # next(all_sequences)
    # for each line except the first, we need to recover the alignment
    count = 0
    for line in alignments:
        if count < 5:
            data = line.split(',')
            reconstruct_alignment(data)
            count += 1
        else:
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
# ['SEQ_KEY', 'ALI_ID', 'SEQ_INDEX', 'SEQ_NAME', 'TAX_ID', 'TAX_CHECK', 'ACC_NR', 'FRAC_WILDCARDS_GAPS', 'CHI2_P_VALUE',
# 'CHI2_PASSED', 'EXCLUDED', 'IDENTICAL_TO', 'FREQ_A', 'FREQ_R', 'FREQ_N', 'FREQ_D', 'FREQ_C', 'FREQ_Q', 'FREQ_E',
# 'FREQ_G', 'FREQ_H', 'FREQ_I', 'FREQ_L', 'FREQ_K', 'FREQ_M', 'FREQ_F', 'FREQ_P', 'FREQ_S', 'FREQ_T', 'FREQ_W',
# 'FREQ_Y', 'FREQ_V', 'SEQ\n']

# def recoverAlns(nprocesses):
#     # create ALN_DIR (if it's not existed)
#     if not os.path.exists(DATA_DIR + ALN_DIR):
#         os.mkdir(DATA_DIR + ALN_DIR)
#
#     # retrieve a list of alns
#     # Opening ALN_INFO_TABLE
#     print('{0}/{1}'.format(DATA_DIR, ALN_INFO_TABLE))
#     input_file = open('{0}/{1}'.format(DATA_DIR, ALN_INFO_TABLE), 'r')
#
#     # init a list of alns
#     alns = []
#     line_index = 0
#
#     # read alns one by one
#     for line in input_file:
#         if (line_index):
#             data = line.split("\t")
#             print(data)
#             alns.append([data[1], data[2], data[3]])
#         line_index += 1
#
#     # Closing the input file
#     input_file.close()
#
#     pool = Pool(nprocesses)                         # Create a multiprocessing Pool
#     with tqdm(total=len(alns)) as pbar:
#         for _ in pool.imap_unordered(partial(recoverAnAln), enumerate(alns)):
#             pbar.update()
#
#
# def main():
#     # parse input arguments
#     parser = argparse.ArgumentParser(
#             formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#             argument_default=argparse.SUPPRESS
#         )
#     parser.add_argument('-p', '--nprocesses', type=int, required=False, help='number of processes (default:1)', default=1)
#     parser.add_argument('-aln', '--aln', type=str, required=False, help='Alignment info', default="aa_aln_info_filtered.csv")
#     parser.add_argument('-dna', '--dna', type=str, required=False, help='DNA sequences', default="aa_sequences.csv")
#     args = parser.parse_args()
#
#     # Recover all alns
#     global ALN_INFO_TABLE
#     global DNA_SEQ_TABLE
#     print('Recover all alignments')
#     ALN_INFO_TABLE = args.aln
#     DNA_SEQ_TABLE = args.dna
#     recoverAlns(nprocesses = args.nprocesses)
#
# if __name__ == "__main__":
#     main()