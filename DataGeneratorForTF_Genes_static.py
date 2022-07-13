import pandas as pd
from numpy import *
import json, re,os, sys
import matplotlib.pyplot as plt

import numpy as np
def get_gene_list_bulk(file_name):
    import re
    h={}
    s = open(file_name,'r',encoding="utf-8")   #gene symbol ID list of bulk RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)',line)
        h[search_result.group(1).lower()]=search_result.group(2)   # h [gene symbol] = gene ID
    s.close()
    return h

def get_gene_list(file_name):
    import re
    h={}
    s = open(file_name,'r') #gene symbol ID list of sc RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)',line)
        h[search_result.group(1).lower()]=search_result.group(2) # h [gene symbol] = gene ID
    s.close()
    return h

def get_sepration_index (file_name):
    import numpy as np
    index_list = []
    s = open(file_name, 'r')
    for line in s:
        index_list.append(int(line))
    return (np.array(index_list))

gap = 100
for name in ["02bone","03dendritic"]: #"01mESC",
    if(name=="03dendritic"):
        sc_gene_list_dir ="data/sc_gene_list.txt"
        mesc_gene_pairs_400_dir = "data/dendritic_gene_pairs_200.txt"
        mesc_gene_pairs_400_num = "data/dendritic_gene_pairs_200_num.txt"

        mesc_cell_expression = "geneData/dendritic_cell.h5"
    if(name=="02bone"):
        sc_gene_list_dir ="data/sc_gene_list.txt"
        mesc_gene_pairs_400_dir = "data/bone_marrow_gene_pairs_200.txt"
        mesc_gene_pairs_400_num = "data/bone_marrow_gene_pairs_200_num.txt"

        mesc_cell_expression = "geneData/bone_marrow_cell.h5"

    if(name=="01mESC"):
        sc_gene_list_dir ="data/mesc_sc_gene_list.txt"
        mesc_gene_pairs_400_dir = "data/mesc_gene_pairs_400.txt"
        mesc_gene_pairs_400_num = "data/mesc_gene_pairs_400_num.txt"

        mesc_cell_expression = "geneData/mesc_cell.h5"

    save_dir = "My1/"+name+"/"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    sc_gene_list = get_gene_list(sc_gene_list_dir)
    mesc_cell_expression = pd.HDFStore(mesc_cell_expression)# scRNA-seq expression data
    RPKMs = mesc_cell_expression['RPKMs']
    mesc_cell_expression.close()
    print('read mesc_cell sc RNA-seq expression')

    # ########## generate NEPDF matrix
    gene_pair_label = []
    s=open(mesc_gene_pairs_400_dir)#'mmukegg_new_new_unique_rand_labelx.txt')#)   ### read the gene pair and label file
    for line in s:
        # print("line",line)  zfp57 \t palmd \t 0 \n
        gene_pair_label.append(line)

    gene_pair_label_array = array(gene_pair_label)
    # print("gene_pair_label_array",gene_pair_label_array)


    gene_pair_index = get_sepration_index(mesc_gene_pairs_400_num)#'mmukegg_new_new_unique_rand_labelx_num.npy')#sys.argv[6]) # read file speration index
    s.close()
    # print("gene_pair_index",gene_pair_index)
    #
    # print("mesc_sc_gene_list('pou5f1')",mesc_sc_gene_list['pou5f1'])


    for i in range(len(gene_pair_index)-1):   #### many sperations
        print (i)
        start_index = gene_pair_index[i]
        end_index = gene_pair_index[i+1]
        x = []
        y = []
        z = []
        for gene_pair in gene_pair_label_array[start_index:end_index]:  ## each speration
            separation = gene_pair.split()
            x_gene_name, y_gene_name, label = separation[0], separation[1], separation[2]
            y.append(label)
            z.append(x_gene_name + ',' + y_gene_name)

            if(name == "01mESC"):
                x_tf = log10(RPKMs[sc_gene_list[x_gene_name]]+ 10 ** -2) # ## 43261 means the number of samples in the sc data, we also have one row that is sum of all cells, so the real size is 43262, that is why we use [0:43261]. For TF target prediction or other data, just remove "[0:43261]"
                x_gene = log10(RPKMs[sc_gene_list[y_gene_name]] + 10 ** -2)# For TF target prediction, remove "[0:43261]"
            else:
                x_tf = log10(RPKMs[int(sc_gene_list[x_gene_name])]+ 10 ** -2) # ## 43261 means the number of samples in the sc data, we also have one row that is sum of all cells, so the real size is 43262, that is why we use [0:43261]. For TF target prediction or other data, just remove "[0:43261]"
                x_gene = log10(RPKMs[int(sc_gene_list[y_gene_name])] + 10 ** -2)# For TF target prediction, remove "[0:43261]"
# log10(np.array(expression_record[gene])+10**-2)
    #         gap = 150
            single_tf_list = []
            for k in range(0, len(x_gene), gap):
                feature = []
                a = x_tf[k:k + gap]
                b = x_gene[k:k + gap]

                feature.extend(a)
                feature.extend(b)
                # single_tf_list.append(feature)
                feature = np.asarray(feature)
                # print("feature.shape", feature.shape)
                if (len(feature) == 2 * gap):
                    # print("feature.shape xixihaha", feature.shape)
                    single_tf_list.append(feature)
                sample_sizex = len(RPKMs)
            single_tf_list = np.asarray(single_tf_list)
#             single_tf_list = (log10(single_tf_list / sample_sizex + 10 ** -3) + 3) / 3
            x.append(single_tf_list)

        save(save_dir + '/Nxdata_tf' + str(i) + '.npy', x)
        save(save_dir + '/ydata_tf' + str(i) + '.npy', array(y))
        save(save_dir + '/zdata_tf' + str(i) + '.npy', array(z))

