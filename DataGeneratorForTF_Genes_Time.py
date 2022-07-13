import numpy as np
import pandas as pd
from numpy import *
import json, re,os, sys

def get_gene_list_bulk(file_name):
    import re
    h={}
    s = open(file_name,'r')   #gene symbol ID list of bulk RNA-seq
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

import time
start_time = time.time()


gap=100
for j in ['01h','02h','03m','04m']:
    if(j == '01h'):
        time_num = 5
        pre = "hesc1"
    if(j == '02h'):
        time_num = 6
        pre = "hesc2"
    if(j == '03m'):
        time_num = 9
        pre = "mesc1"
    if(j == '04m'):
        time_num = 4
        pre = "mesc2"

    a_path = "../Database/input/"+pre+"_gene_list_ref.txt"
    b_path = "../Dataset/input/scRNA_expression_data/"+pre+"_expression_data/"
    c_path = "../Dataset/input/"+pre+"_gene_pairs_400.txt"
    d_path = "../Dataset/input/"+pre+"_gene_pairs_400_num.txt"


    save_dir = "../Dataset/TwoNetwork/My/"+j+"/"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)


    h_gene_list = a_path
    sample_size_list = []
    sample_sizex = []
    total_RPKM_list = []
    gene_expression_path = b_path

    for indexy in range (0,time_num):

        if(j == '03m'):
            store = pd.HDFStore(gene_expression_path+'RPM_'+str(indexy)+'.h5')#    # scRNA-seq expression data                        )#
            rpkm = store['/RPKM']
        else:
            store = pd.HDFStore(gene_expression_path+'RPKM_'+str(indexy)+'.h5')#    # scRNA-seq expression data                        )#
            rpkm = store['RPKMs']

        store.close()
        total_RPKM_list.append(rpkm)
        sample_size_list = sample_size_list + [indexy for i in range (rpkm.shape[0])] #append(rpkm.shape[0])
        sample_sizex.append(rpkm.shape[0])
        samples = array(sample_size_list)
        sample_size = len(sample_size_list)


    total_RPKM = pd.concat(total_RPKM_list, ignore_index=True)


    ########## generate NEPDF matrix
    gene_pair_label = []
    label_path = c_path
    s=open(label_path) ### read the gene pair and label file
    for line in s:
        gene_pair_label.append(line)
    label_num_path=d_path
    gene_pair_index = get_sepration_index(label_num_path)#'mmukegg_new_new_unique_rand_labelx_num.npy')#sys.argv[6]) # read file speration index
    s.close()
    gene_pair_label_array = array(gene_pair_label)
    for i in range(len(gene_pair_index)-1):   #### many sperations
#         print (i)
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

            x_tf = log10(total_RPKM[x_gene_name] + 10 ** -2)
            x_gene = log10(total_RPKM[y_gene_name] + 10 ** -2)
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
                sample_sizex = len(total_RPKM)
            single_tf_list = np.asarray(single_tf_list)

#             single_tf_list = (np.log10(single_tf_list / sample_sizex + 10 ** -3) + 3) / 3
            x.append(single_tf_list)

        save(save_dir + '/Nxdata_tf' + str(i) + '.npy', x)
        save(save_dir + '/ydata_tf' + str(i) + '.npy', array(y))
        save(save_dir + '/zdata_tf' + str(i) + '.npy', array(z))
    end_time = time.time()
    all_time = end_time - start_time
    print(j+":all_time:"+str(all_time))
    f = open("My_time.txt",mode="a")
    f.writelines(j+":all_time:"+str(all_time)+"\n")
    f.close()

