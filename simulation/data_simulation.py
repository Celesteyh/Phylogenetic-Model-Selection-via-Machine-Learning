# generate command line for IQ-TREE2 AliSim
# iqtree2 --alisim file_name -t RANDOM{yh/n_taxa} --branch-distribution branch_length -m model_name+F{frequencies}+I{invariant}+G4{alpha}+Rn{w1,r1,...} --distribution custom_distributions.txt

import sys
import json
from scipy import stats
import os
from ete3 import Tree
import time

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


# distributions = {'external': ['johnsonsu', (-2.3814141699162743, 0.5626377101339911, 0.0008975988787962728, 0.0004482087535380683)],
#                  'internal': ['invweibull', (0.8991407952200723, -0.0007145104346082998, 0.006685532702753417)],
#                  'invariant': ['genexpon', (0.6367621010042517, 2.57737326480264, 0.7327418951929136, 0.0010999999897400845, 0.3476884498414342)],
#                  'alpha': ['invweibull', (4.250986436713845, -1.2026325157923896, 2.0112207861825153)],
#                  'R2_w1': ['johnsonsu', (0.9112115964902836, 0.7163315070511832, 0.9399061778116748, 0.02402239429078332)],
#                  'R2_w2': ['loglaplace', (1.5761453478225333, 0.002592059597768499, 0.09880776103141849)],
#                  'R2_r1': ['genlogistic', (3.012299702290826, 0.24280135780415796, 0.09366843353787556)],
#                  'R2_r2': ['johnsonsu', (-0.4663805751995879, 0.9476841984987253, 5.239153516034693, 2.3830414012046357)],
#                  'R3_w1': ['johnsonsu', (1.2328125021975276, 1.2213930669799662, 0.8078536183792369, 0.0819150349926418)],
#                  'R3_w2': ['johnsonsu', (-1.7217926003078725, 1.8175842264177882, 0.12498526742800138, 0.09325077709029117)],
#                  'R3_w3': ['loglaplace', (1.7500668981114917, 0.0023541105721471773, 0.058868525498502924)],
#                  'R3_r1': ['johnsonsu', (-1.3638735496950933, 2.032113954170627, 0.10712081091678177, 0.08643161632056487)],
#                  'R3_r2': ['johnsonsu', (-0.7124415048052484, 1.5875000694688282, 1.5157104240934969, 0.9074115107528593)],
#                  'R3_r3': ['johnsonsu', (-0.7071229756249926, 1.1224922736815843, 5.2786532755221955, 2.5241431206769818)],
#                  'R4_w1': ['genlogistic', (0.3074404672073501, 0.5943955308140962, 0.04704086064879576)],
#                  'R4_w2': ['johnsonsu', (-0.5346975573258534, 2.301828245142554, 0.23116516684566663, 0.12704351044159803)],
#                  'R4_w3': ['exponnorm', (3.1388906149807774, 0.10836384454339926, 0.03203839172994989)],
#                  'R4_w4': ['alpha', (2.6770807971750226, -0.04444846061736995, 0.24059159259074586)],
#                  'R4_r1': ['alpha', (8.239303608958101, -0.1712365706550642, 2.2118355252814297)],
#                  'R4_r2': ['genlogistic', (4.066523235594884, 0.46621437563582074, 0.20987051579568983)],
#                  'R4_r3': ['exponnorm', (1.4172642831462268, 1.7077548854870712, 0.6233741931199277)],
#                  'R4_r4': ['exponnorm', (2.227728274088503, 3.2770189552103774, 1.3682428938283921)]}

# save the dictionary to a file
# with open('distributions.json', 'w') as f:
#     json.dump(distributions, f)

# # save the dictionary to a file
# with open('frequencies.json', 'w') as f:
#      json.dump(F, f)

# np.random.seed(6)

class_mapping = {
        'LG': 0, 'WAG': 1, 'JTT': 2,
        'Q.plant': 3, 'Q.bird': 4,
        'Q.mammal': 5, 'Q.pfam': 6
    }


def sample_from_distribution(dist_name, dist_params, lbound, ubound, r):
    sample = round(getattr(stats, dist_name).rvs(*dist_params), r)
    while sample <= lbound or sample >= ubound:
        sample = round(getattr(stats, dist_name).rvs(*dist_params), r)
    return sample


def generate_command_line(model_name, F, I, G4, Rn, n_taxa, length, iteration):
    # iqtree2 --alisim file_name -t tree.nwk -m model_name+F{frequencies}+I{invariant}+G4{alpha}+Rn{w1,r1,...}
    if (model_name not in ['LG', 'WAG', 'JTT', 'Q.plant', 'Q.bird', 'Q.mammal', 'Q.pfam']) or (G4 and Rn):
        print("Invalid input")
        sys.exit(0)

    para_dist = json.load(open('distributions.json'))
    freq_dist = json.load(open('frequencies.json'))

    directory = 'E:\\PycharmProjects\\Phylogenetics'

    # use an arbitrary model to generate a random tree
    command_tree = 'E:\\iqtree-2.2.2.6-Windows\\iqtree-2.2.2.6-Windows\\bin\\iqtree2 --alisim gen_tree -t RANDOM{yh/' + str(n_taxa) + '} -m LG -redo'
    os.system('e: && cd ' + directory + '\\trees_more && ' + command_tree)

    # modify the branch lengths of the tree
    tree_file = directory + '\\trees_more\\gen_tree.treefile'
    tree = open(tree_file, 'r').read()
    t = Tree(tree)
    for node in t.traverse():
        if not node.is_root():
            if node.is_leaf():
                node.dist = sample_from_distribution(para_dist['external'][0], para_dist['external'][1], 0, 10, r=10)
            else:
                node.dist = sample_from_distribution(para_dist['internal'][0], para_dist['internal'][1], 0, 10, r=10)

    # command = f"E:\\iqtree-2.2.2.6-Windows\\iqtree-2.2.2.6-Windows\\bin\\iqtree2 --alisim {file_name} -t {new_tree_file} -m {model_name}"
    # TODO: change this
    file_name = f"{class_mapping[model_name]}({n_taxa}){model_name}"  # v for validation; t for test
    command = f"-m {model_name}"

    if F:
        # sample 20 frequencies from the distribution
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
        invariant = sample_from_distribution(dist_name, dist_params, 0, 0.9, r=5)  # min in data = 0.0011
        command += '+I{' + str(invariant) + '}'
        file_name += '+I{' + str(invariant) + '}'

    if G4:
        dist_name = para_dist['alpha'][0]
        dist_params = para_dist['alpha'][1]
        alpha = sample_from_distribution(dist_name, dist_params, 0.001, 11, r=5)  # min in data = 0.02; max = 124.8781, second = 10.7034
        command += '+G4{' + str(alpha) + '}'
        file_name += '+G4{' + str(alpha) + '}'

    elif Rn != 0:
        w_r = []
        for i in range(1, Rn + 1):
            w_name = 'R' + str(Rn) + '_w' + str(i)
            r_name = 'R' + str(Rn) + '_r' + str(i)
            w = sample_from_distribution(para_dist[w_name][0], para_dist[w_name][1], 0, 1, r=5)
            # min for rates = 0.0361 1.0107 0.0004 0.0981 1.1293 0.0016 0.0723 0.2221 1.2137
            r = sample_from_distribution(para_dist[r_name][0], para_dist[r_name][1], 0.0001, 200, r=5)
            w_r.append((w, r))
        command += '+R' + str(Rn) + '{' + ','.join([str(w) + ',' + str(r) for w, r in w_r]) + '}'
        file_name += '+R' + str(Rn)
    # TODO: change this
    # file_name += f'[{length}]'  # only for testing data
    file_name += f'_{iteration + 300}'

    # save the new tree with the model name
    new_tree = t.write(format=1)
    new_tree_file = directory + f'\\trees_more\\{file_name}.treefile'
    with open(new_tree_file, 'w+') as result:
        result.write(new_tree + '\n')
    command = 'E:\\iqtree-2.2.2.6-Windows\\iqtree-2.2.2.6-Windows\\bin\\iqtree2 --alisim ' + file_name + f' -t {new_tree_file} ' + command + ' --length ' + str(length)
    print(command)
    os.system('e: && cd ' + directory + '\\simulation_more && ' + command)
    return


def generate_all_models(base_model_list, num_iterations):
    """
    Generate all models for each base model, number of taxa, and iteration
    """
    for i in range(num_iterations):
        for base_model in base_model_list:
            # for testing data
            # for length in [100, 1000, 10000, 100000]:
            for n_taxa in [8, 16, 32, 64, 128]:
                for F in [True, False]:
                    for I in [True, False]:
                        # G4 and Rn cannot appear together
                        for G4 in [True, False]:
                            if G4:
                                generate_command_line(base_model, F, I, True, 0, n_taxa, 1000, i)
                            else:
                                for Rn in [0, 2, 3, 4]:
                                    generate_command_line(base_model, F, I, False, Rn, n_taxa, 1000, i)


if __name__ == '__main__':
    start_time = time.time()
    # 7 base models: LG, WAG, JTT, Q.bird, Q.mammal, Q.plant, Q.pfam
    # +F, +I, +G4/+R2/+R3/+R4
    # n_taxa = 8, 16, 32, 64, 128
    # length = 1000
    # 18s 100 files and 5Mb per base_model per replicate
    # 5Mb * 11 * 100 = 5500Mb = 5.5Gb
    # 100 files * 11 * 100 = 110,000 files
    # 18s * 11 * 100 = 19,800s = 5.5 hours
    base_model_list = ['LG', 'WAG', 'JTT', 'Q.plant', 'Q.bird', 'Q.mammal', 'Q.pfam']
    # base_model_list = ['LG']
    # generate 10 alignments for each category

    # test: length = 100, 1k, 10k, 100k; taxa = 8, 16, 32, 64, 128, 256, 512, 1024
    # total combinations: 7 * 2 * 2 * 5 * 4 * 5 + 7 * 2 * 2 * 5 * 2 = 2800 + 280 = 3080
    # 30 replicates: 3080 * 30 = 92,400
    generate_all_models(base_model_list, 50)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # generate_command_line('test6', 'LG', True, False, False, 0, 4, 100)
    # a1 = 'KWKLTGLVHLLHGATVLQISDKLAGQEGERFSLNNFETRCDPESNADRALMSLALKIKHNALRTYAEDDPNGLAWDRERETNMQLCWSWFVQKIHQILRYKWKLTGLVHLLHGATVLQISDKLAGQEGERFRLNNFETRCDPQSNADRALASLALKIKHNALRTYAEDDPNGLAWDRERETNMQLCWVWFVQKIHQILRYKWKLTGLVHLLHGATVLQISDKLAGQEGERFRLNNFETRCDPESNADRALMSLALKIKHNALRTYAEDDPNGLAWDRERETNMQLCWSWFVQKIHQILRYKWKLTGLVHLLHGATVLQISDKMAGQTGERFRLNNFETRCDPESNANRALMSLALKINHNALRTYAEDDENGLAWDRERITNMQLCWSWFVQKIHQILRY'
    # a2 = 'VRALEELRSLLQLIMGTRQTTIRIEFASADDTVNAAQLNDFQREVPKLRLIMLRSLLLLILGYAYNSIDYENHRYYFLDEVKPQALSLISRFIAGLYFRTVRALEELRSLLQLIMGTRQTTLRIEFASADDTVNAAQLNDFQREVPKLRLIMLRSLLLLILGYAYNSIDYENHRYYFLDEVKPQALSLISRFIAGLYFRTVRALEELRSLLQLIMGTRQTTLRIEFASADDTVNAAQLNDFQREVPKLRLIMLRSLLLLILGYAYNSIDYENHRYYFLDEVKPQALSLISRFIAGLYFRTVRALEELRSLLQLIMGTRQTTLRIEFASADDTVNAAQLNDFQREVPKLRLIMLRSLLLLILGYAYNSIDYENHRYYFLDEVKPQALSLISRFIAGLYFRT'
    # a3 = 'MLADVHNLGVGAPIVALMAKCALIENAYERELAMIPYYNRCAHGDARFGDRGRFSLKQERPLKHHKYWPYQGSPILRAVSMKFILLIAWNVRGTKRYADLMLADVHNLGVGAPIVALMAKCALIENAYERELAMIPYYNRCAHGDARFGDRGRFSLKQERPLKHHKYWPYQGSPILRAVSMKFILLIAWNVRGTKRYADLMLADVHNLGVGAPIVALMAKCALIENAYERELAMIPYYNRCAHGDARFGDRGRFSLKQERPLKHHKYWPYQGSPILRAVSMKFILLIAWNVRGTKRYADLMLAEVHNLGVGAPIVALMAKCALIENAYERELAMIPYYNKCAHGDARFGDRGRFSLKQERPLKHHKYWPYQGSPILRAVSMKFILLIAWNVRGTKRYADL'
    # a4 = 'GIRVHKSPYRNLVTPGPGWARWASFHSLLRIVLPLKRRIKSLLDLEKFDGASTFTSKLSGLELYTLSVNFYGPLEFLAPEKIRSGKVWPTLEAPEMKIIAGIRVHKTPYRNLVSPGPGWARWASFHSLLRVVLPLKRRIKSLLDLEKFDGASTFTSKLSGLGLYTLSVNFYGPLEFLAPEKIRSGKIWPTLEAPEMKIIAGLRIHKRPYRGLITPGRGWARWATAHSLFSVVFPLKWRLKSFCELVRFDGASTFNSKLSGLQLYGLDVNWFGPLEIMPPEDIKAGKAWPPLEAPEMKLIAGISVHKSPYRNLVTPGPGWARWASFHSLFRVVLPLKRRIKSLLDLEKFDGAATFTSKISGLELYELSVNFYGPLEFLAPEKIRSGKVWPTLEAPEMKIIA'
    # a5 = 'PEVLILGRLPDANARIRNLEIMLTSEGTGGEAIVDREVRETMHFIVKLYRIENVNTLTMIDDENIDNVKLLFALKVIRHFPLSLLTVGNIFKLTQDKVQFPEVLIIGRLPDANARIRNLEIMLTSQGTGGESIVDREVRKTMHFIVKMYCIENVNTLTMIDTENIDNVKMLFALKVLRHFPLSLLTVGNIFMLTQDKVQFPEVLILGRLSDANARIRNLEIMLTSEGTGGEAIVDREIRETMHFIVKLYKIENVNTLTLIDDENIDNVKLLFALKVIRHFPLSLLTVGNIFKLTQDKVQFPEVLILGRLPDANARIRNLEIMLTSEGTGGEAIVDREVRETMHFIVKLYKIENVNTLTLIDGENIDNVKLLFALKVIRHFPLSLLTVGNIFKLTQDKVQF'
    # aa6 = 'EEARTDRDKPRLDLSLALEVGIERLFDFGFLTKTVSTSPKTIYRDEGYANKNAYTRDLEKRSCTAGSKKTARNRVLIARKVKLTEPESGKPLEEESGTNGEEARTDRDKPRLDLSLALEVGIERLFDFGFLTKTVSTSPKTIYRDEGYANKNAYTRDLEKRSCTAGSKKTARNRVLIARKVKLTEPESGKPLEEESGTNGSEFRTDKDDPRIDLRLGLEVGIERLFDFGLLTKTVSTAPKTIDREEAFANRKRYTTDLDQRSCSAGNKTGAKKRLLVSRKVKLTEPNSAKSLEEESGRGGNELRSDRDEPRLDLRLALEVGIERLFDFGFLTKTVSTSPKSIYRDEAYANRKRYTRDLEKRSCTAGSRNSARTRVLIAKKVKLTEPESGKPLEEESGTNG'
    # aa7 = 'PYPAEANLSTAGTGTSKPEIAWPKGLLGEGGREQDPRGKQFKPRVRSKGSPLNPYGRDTHHRYPFFPSPKGQQKQVSVEGRGRLSPCGQLTSRLTYHETGAGTNLSPAGTGTSKPEHAWSKPLLEGHGREQDPRGKCFKPRVRSKGSPLNPVGRQTHGRYPFPPSSMGQQKQVSVEGRGRLSPQGQLTSRLTYLETGKEQAGANLSPAGTGTSKPEHAWSKHLLEGHGREQDPRGKCFKPRVRSKGSPLNPVGRQTHGRYPFPPSSMGQQKQVSVEGRGRLSPQGQLTSRLTYLETGKEQPYPAETNLSPAGTGTSKSEIAWPKGLLGESGREQDSRGKQMKPRVRSKGSPLNPYGRDTHHRYPLFPSPKGQQKQVSVEGRGRLSPCGQLSSRLSYHETG'
    # aa8 = 'KEAGKPSYFLSSSSAALPSSENELKLMDDDANDDEDMMMKNSKNDSLGRNAESNEKSWKHKDSERKAPHAMTSDSAKRKPKAQMKENLNEMQPKADAADEKKAGKPPYFLSSSSAALPSSENELKQMDDDANDDEDMMMKNSKNDSLGRNAESNEKSWKHRDSERKAPHAMTSDSAKRKPKAQVKEHLNEMQPRAEAAEEKKAGNPPYFLSSSSAALPSSENELKQMDDDANDDEDRMMKNSKNDSMGRNAESNEKSWKHRDSERKAPHAMTSDSAKRMPKAQVKEHLNEMQPKAEAAEEKKAGKPPYFLSSSSAALPSSENELKQMDDDANDDEDMMMKNSKNDSLGRNAESNEKSWKHRDSERKAPHAMTSDSAKRKPKAQVKEHLNEMQPKAEAAEE'
    # aa9 = 'ERERCTEAAPRRPILEASCLTRRSEQQSTEFQSSGRLTLREDGCCAPTLNQQLRWARAELTLQLERTRRSKMLPNGWPPAYSISDRHEIPRGVAVLTGASERERCTEAAPRRPILEASCLTRRSEQQSTEFQSSGRLTLREDGCCAPTLNQQLRWARAELTLQLERTRRSKMLPNAWPPAYSISDRHEIPRGVAVLTGASERERCTEAAPRRPILEASCLTRRSEQQSTEFQSSGRLTLREDGCCAPTLNQQLRWARGELTLQLERTRRSKMLPNGWPPAYSISDRHEIPRGVAVLTGASARERCTEAAPRRPILEASCLTRRSEQQSTEFQSSGRLRLREDGCCAPSLNQQLRWARAELTLQLERTRRSKILPNGWPPAYSISERHEIPRGVAVLTGAS'
    # # aa = 'VFSSKAVISEGTMPPAPSVPSKQQFDYAMETKNEFPIIVFIQMRNDQAIHAPDAENPGKTDVSENTKEVQKRSWSYMSNSQQWSEAFQAQDFAADFLKNSVFSSKAIISEGTMPPAPSAPSKQQFDYAMETKNEFPIIAFIRTRNDQAIHFPEAENPGKTDVQEHTKEVQKRSWSFLSNRQAWAQAFQAQAFAADFLKNPVFSSKAIIEEGTMPPAPSAPSKQQFDYAMETKNEFPIIAFIRTRNDQCIHFPEAEDPGKTDVQEHTKEVQKSSWSYLSNRQAWAQAFQAQAFAADFLKNPVFSSRAIISEGSMPPAPSAPSKQQFDYAMETKREFQIIAFIRTRNDQAIHFPEAENPGKTSVQEHTKEVQKRSWSYLSNRQAWAQAYRAQAFAADFLKNSVFSSKAVISEGTMPPAPSVPSKQQFDYAMETKNEFPIIVFIQTRNDQAIHAPDAENPGKTDVSENTKEVQKRSWSYISNSQQWSEAFQAQAFAADFLKNSVFSSKAVISEGTMPPAPSVPSKQQFDYAMETKNEFPIIVFIQTRNDQAIHAPDAENPGKTDVSENTKEVQKRSWSFMSNSQQWSEAFQAQAFAADFLKNSVFSSKAVISEGTMPPAPSVPSKQQFDYAMETKNEFPIIVFIQTRNDQAIHAPDAENPGKTDVSENTKEVQKRSWSFMSNSQQWSEAFQAQAFAADFLKNSVFSTKAVISEGTMPPAPSVPSKQQFDYAMETKKDFPIIVFVQTRNEQAINAPDAENPGKTNVSERTQEIQKRSWPSMSNSPKWSEAFTAQAFAADFLENS'
    # # f = [0.07435,0.04426,0.0329,0.04622,0.00569,0.03806,.03549,0.02377,0.01732,0.04112,0.04788,0.06042,0.02743,0.07256,0.06576,0.06262,0.03083,0.0032010534511040176,0.0185,0.04761]
    # # f = (f / np.sum(f)).round(5)
    # # new_f = [0.09341, 0.0556,  0.04133, 0.05807, 0.00715, 0.04781, 0.04459, 0.02986, 0.02176, 0.05166, 0.06015, 0.07591, 0.03446, 0.09116, 0.08261, 0.07867, 0.03873, 0.00402, 0.02324, 0.05981]
    # # print(f.sum())
    # # # check the frequencies of the amino acids in all the sequences
    # for a in 'ARNDCQEGHILKMFPSTWYV':
    #     # use comma as separator
    #     print(aa9.count(a)/400, end=', ')
    # amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    # set1_freq = [0.0825, 0.0775, 0.065, 0.0575, 0.02, 0.0525, 0.065, 0.05, 0.04, 0.0425, 0.1375, 0.0575, 0.02, 0.03,
    #              0.0175, 0.04, 0.0525, 0.04, 0.02, 0.0325]
    # set2_freq = [0.08, 0.1, 0.04, 0.05, 0.0, 0.05, 0.06, 0.03, 0.01, 0.0725, 0.1875, 0.02, 0.02, 0.05, 0.02, 0.06, 0.05,
    #              0.0, 0.06, 0.04]
    # set3_freq = [0.12, 0.0875, 0.04, 0.0375, 0.02, 0.02, 0.0425, 0.07, 0.04, 0.06, 0.11, 0.0625, 0.04, 0.03, 0.05, 0.03,
    #              0.01, 0.02, 0.06, 0.05]
    # set4_freq = [0.0675, 0.0675, 0.02, 0.0225, 0.0025, 0.0025, 0.0575, 0.0775, 0.02, 0.0575, 0.14, 0.0775, 0.0125, 0.055,
    #              0.0825, 0.0825, 0.0425, 0.035, 0.0275, 0.05]
    # set5_freq = [0.0375, 0.0625, 0.07, 0.055, 0.0025, 0.0225, 0.075, 0.0525, 0.02, 0.1025, 0.14, 0.055, 0.0325, 0.05,
    #              0.0275, 0.025, 0.0725, 0.0, 0.01, 0.0875]
    # set6_freq = [0.065, 0.11, 0.0375, 0.065, 0.01, 0.0025, 0.105, 0.07, 0.0, 0.03, 0.105, 0.095, 0.0, 0.0325, 0.0375,
    #              0.0775, 0.0925, 0.0, 0.025, 0.04]
    # set7_freq = [0.035, 0.09, 0.02, 0.015, 0.01, 0.07, 0.06, 0.1375, 0.0325, 0.005, 0.0775, 0.07, 0.0075, 0.02, 0.11,
    #              0.1025, 0.0625, 0.01, 0.03, 0.035]
    # set8_freq = [0.12, 0.0425, 0.075, 0.095, 0.0, 0.0275, 0.1075, 0.02, 0.0275, 0.0, 0.05, 0.1325, 0.065, 0.01, 0.0575,
    #              0.1325, 0.01, 0.01, 0.01, 0.0075]
    # set9_freq = [0.0925, 0.1525, 0.02, 0.0175, 0.04, 0.06, 0.1, 0.05, 0.01, 0.0325, 0.11, 0.01, 0.0075, 0.01, 0.07,
    #              0.0925, 0.075, 0.02, 0.01, 0.02]
    # # creating the plot
    # plt.figure(figsize=[12, 10])
    # plt.plot(amino_acids, set1_freq, label='Set 1', color='r', alpha=0.7)
    # plt.plot(amino_acids, set2_freq, label='Set 2', color='r', alpha=0.7)
    # plt.plot(amino_acids, set3_freq, label='Set 3', color='r', alpha=0.7)
    # plt.plot(amino_acids, set4_freq, label='Set 4', color='r', alpha=0.7)
    # plt.plot(amino_acids, set5_freq, label='Set 5', color='r', alpha=0.7)
    # # plt.plot(amino_acids, set6_freq, label='Set 6', color='orange')
    # # plt.plot(amino_acids, set7_freq, label='Set 7', color='orange')
    # # plt.plot(amino_acids, set8_freq, label='Set 8', color='orange')
    # plt.plot(amino_acids, set9_freq, label='Set 9', color='orange')
    # plt.title('Amino Acid Frequencies')
    # plt.xlabel('Amino Acid')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
