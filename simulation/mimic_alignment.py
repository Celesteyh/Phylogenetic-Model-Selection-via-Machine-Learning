import os
import time
import shutil
import sys
import random

alignments = 'E:/PycharmProjects/Phylogenetics/aln/real_clean_2'
treefiles = 'E:/PycharmProjects/Phylogenetics/real_treefiles'
outpath = 'E:/PycharmProjects/Phylogenetics/balanced_mimicked'

model_map = {'0': 'LG', '1': 'WAG', '2': 'JTT', '3': 'Q.plant', '4': 'Q.bird', '5': 'Q.mammal', '6': 'Q.pfam'}

start_time = time.time()

# path = 'E:/PycharmProjects/Phylogenetics/mimicked_alignments'
# l = []
# for alignment in os.listdir(path):
#     if alignment.endswith('.log'):
#         if not os.path.isfile(os.path.join(path, alignment.split('.')[0] + '.phy_1.phy')):
#             l.append(alignment[:-4])

# error_list = []
# for file in os.listdir(outpath):
#     if file.endswith('.log'):
#         if not os.path.isfile(os.path.join(outpath, file.split('.')[0] + '.phy_1.phy')):
#             error_list.append(file)
# print(len(error_list))
# print(error_list)
# # ['3ENSG00000101421_CHMP4B+R2.phy.log', '3ENSG00000104879_CKM+I+G4.phy.log', '3ENSG00000106992_AK1+I+G4.phy.log', '3ENSG00000120053_GOT1+R3.phy.log', '3ENSG00000130713_EXOSC2+I+G4.phy.log', '3ENSG00000137563_GGH+I+G4.phy.log', '3ENSG00000149100_EIF3M+R2.phy.log', '3ENSG00000155097_ATP6V1C1+R2.phy.log', '3ENSG00000164879_CA3+R3.phy.log', '3ENSG00000167130_DOLPP1+I+G4.phy.log', '3ENSG00000167460_TPM4+F+R3.phy.log', '3ENSG00000170312_CDK1+R2.phy.log']
# # ['4ENSG00000027847_B4GALT7+I+G4.phy.log', '4ENSG00000099942_CRKL+I+G4.phy.log', '4ENSG00000101337_TM9SF4+I+G4.phy.log', '4ENSG00000127554_GFER+I+G4.phy.log', '4ENSG00000128045_RASL11B+R3.phy.log', '4ENSG00000135373_EHF+I+G4.phy.log', '4ENSG00000145391_SETD7+R3.phy.log', '4ENSG00000155868_MED7+I+G4.phy.log', '4ENSG00000179115_FARSA+R4.phy.log', '4ENSG00000180697_C3orf22+I+G4.phy.log', '5ENSG00000103510_KAT8+I+G4.phy.log', '5ENSG00000106689_LHX2+R2.phy.log', '5ENSG00000108924_HLF+I+G4.phy.log', '5ENSG00000113597_TRAPPC13+R2.phy.log', '5ENSG00000128710_HOXD10+R2.phy.log', '5ENSG00000130822_PNCK+R3.phy.log', '5ENSG00000149480_MTA2+R3.phy.log', '5ENSG00000176101_SSNA1+R2.phy.log', '5ENSG00000181722_ZBTB20+I+G4.phy.log', '5ENSG00000214517_PPME1+R2.phy.log']
# sys.exit()

# selected_6 = 'E:\\PycharmProjects\\Phylogenetics\\aln\\selected_6'
# # for alignment in os.listdir(alignments):
# #     if alignment[0] == '6':
# #         shutil.copyfile(os.path.join(alignments, alignment), os.path.join(selected_6, alignment))
# to_remove = random.sample(os.listdir(selected_6), 839)
# for file in to_remove:
#     os.remove(os.path.join(selected_6, file))

# el = ['5ENSG00000103510_KAT8+I+G4.phy.log', '5ENSG00000106689_LHX2+R2.phy.log', '5ENSG00000108924_HLF+I+G4.phy.log', '5ENSG00000113597_TRAPPC13+R2.phy.log', '5ENSG00000128710_HOXD10+R2.phy.log', '5ENSG00000130822_PNCK+R3.phy.log', '5ENSG00000149480_MTA2+R3.phy.log', '5ENSG00000176101_SSNA1+R2.phy.log', '5ENSG00000181722_ZBTB20+I+G4.phy.log', '5ENSG00000214517_PPME1+R2.phy.log']
# for error_aln in el:
#     os.remove(os.path.join(outpath, error_aln))
#     print(f"Removed {error_aln}")
# sys.exit()

for alignment in os.listdir('E:\\PycharmProjects\\Phylogenetics\\aln\\selected_5'):
# for error_aln in error_list:
#     os.remove(os.path.join(outpath, error_aln))
#     alignment = error_aln[:-4]
# for alignment in l:
    # os.remove(os.path.join(outpath, alignment + '.log'))
    # if it doesn't exist in the outpath
    if not os.path.isfile(f"{os.path.join(outpath, alignment)}.log"):
    # if True:
        # open the alignment file and replace '?', '.', '~', '*', '!' or 'X' with '-' to avoid errors
        # then save it to a new file
        # with open(os.path.join(alignments, alignment), 'r') as file:
        #     content = file.read()
        #     if '!' in content:
        #         # replace '!' with '-'
        #         new_content = content.replace('!', '-')
        #         with open(os.path.join(alignments, alignment), 'w') as file_2:
        #             file_2.write(new_content)

        treefile = os.path.join(treefiles, f"{alignment.split('.')[0]}.treefile")
        # alignment = '0PF00176+F+I+G4.phy'  '0ENSG00000070831_CDC42+.phy'
        addition = alignment.split('.')[0].split('+')
        # addition = ['0ENSG00000070831_CDC42', '']  ['0PF00176', 'F', 'I', 'G4']
        if addition[1] != '':
            model = f"{model_map[alignment[0]]}+{'+'.join(addition[1:])}"
        else:
            model = model_map[alignment[0]]
            # TODO: -keep-ident
        # new version of iqtree
        command = f"E:\\iqtree-2.3.4-Windows\\iqtree-2.3.4-Windows\\bin\\iqtree2 --alisim {os.path.join(outpath, alignment)} -s {os.path.join(alignments, alignment)} -te {treefile} -m {model} --prefix {os.path.join(outpath, alignment)} --num-alignments 115 -nt 7"
        # print(command)
        os.system(command)
        # remove unnecessary files if exist
        try:
            os.remove(f"{os.path.join(outpath, alignment)}.uniqueseq.phy")
        except FileNotFoundError:
            pass
        try:
            os.remove(f"{os.path.join(outpath, alignment)}.ckp.gz")
            os.remove(f"{os.path.join(outpath, alignment)}.iqtree")
            os.remove(f"{os.path.join(outpath, alignment)}.treefile")
        except Exception as e:
            # stop
            print(f"{alignment} has an error: {e}")
            break
            # file_path = f"{os.path.join(outpath, alignment)}.log"  # 'E:/PycharmProjects/Phylogenetics/mimicked_alignments\\3ENSG00000109814_UGDH+R2.phy.log'
            # name = file_path[:-4]  # 'E:/PycharmProjects/Phylogenetics/mimicked_alignments\\3ENSG00000109814_UGDH+R2.phy'
            # # move the log file to the correct directory
            # dir = 'E:/PycharmProjects/Phylogenetics/excluded_alignments/error'
            # os.rename(file_path, os.path.join(dir, alignment + '.log'))
            # # move the real alignment file to the correct directory
            # os.rename(os.path.join(alignments, alignment), os.path.join(dir, alignment))
end_time = time.time()
print('---------------------------------------------------------------------------------')
print(f"Time taken: {end_time - start_time}")


# def get_line_number(file_path, line_number):
#     with open(file_path, 'r') as file:
#         for current_line_number, line in enumerate(file, start=1):
#             if current_line_number == line_number:
#                 return line
#     return None
#
#
# file_path = 'E:/PycharmProjects/Phylogenetics/aln/real/0ENSG00000087191_PSMC5+.phy'
# line_content = get_line_number(file_path, 104)
# if line_content is not None:
#     print(line_content)
# else:
#     print("Line 104 does not exist in the file.")