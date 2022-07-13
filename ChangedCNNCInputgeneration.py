import csv
import numpy as np
import csv
from numpy import *
import os

def get_tf_list(tf_path):
    # return tf_list
    f_tf = open(tf_path)
    tf_reader = list(csv.reader(f_tf))
    tf_list=[]
    for single in tf_reader[1:]:
        tf_list.append(single[0])
    print('Load '+str(len(tf_list))+' TFs successfully!')
    return tf_list

def get_origin_expression_data(gene_expression_path):
    # return 1.tf-targets dict and pair-score dict
    #        2.number of timepoints
    f_expression = open(gene_expression_path,encoding="utf-8")
    expression_reader = list(csv.reader(f_expression))
    cells = expression_reader[0][1:]
    num_cells = len(cells)

    expression_record = {}
    num_genes = 0
    for single_expression_reader in expression_reader[1:]:
        if single_expression_reader[0] in expression_record:
            print('Gene name '+single_expression_reader[0]+' repeat!')
        expression_record[single_expression_reader[0]] = list(map(float, single_expression_reader[1:]))
        num_genes += 1
    print(str(num_genes) + ' genes and ' + str(num_cells) + ' cells are included in origin expression data.')
    return expression_record,cells

def get_normalized_expression_data(gene_expression_path):
    # return 1.tf-targets dict and pair-score dict
    #        2.number of timepoints
    expression_record,cells=get_origin_expression_data(gene_expression_path)
    expression_matrix = np.zeros((len(expression_record), len(cells)))
    index_row=0
    for gene in expression_record:
        expression_record[gene]=np.log10(np.array(expression_record[gene])+10**-2)
        expression_matrix[index_row]=expression_record[gene]
        index_row+=1

    #Heat map
    # plt.figure(figsize=(15,15))
    # sns.heatmap(expression_matrix[0:100,0:100])
    # plt.show()

    return expression_record, cells

def get_gene_ranking(gene_order_path,low_express_gene_list,gene_num,output_path,flag):#flag=True:write to output_path
    #1.delete genes p-value>=0.01
    #2.delete genes with low expression
    #3.rank genes in descending order of variance
    #4.return gene names list of top genes and variance_record of p-value<0.01
    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    if flag:
        f_rank = open(output_path, 'w', newline='\n')
        f_rank_writer = csv.writer(f_rank)
    variance_record = {}
    variance_list = []
    significant_gene_list=[]
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        if float(single_order_reader[1]) >= 0.01:
            break
        if single_order_reader[0] in low_express_gene_list:
            continue
        variance = float(single_order_reader[2])
        if variance not in variance_record:# 1 variance corresponding to 1 gene
            variance_record[variance] = single_order_reader[0]
        else:# 1 variance corresponding to n genes
            print(str(variance_record[variance]) + ' and ' + single_order_reader[0] + ' variance repeat!')
            variance_record[variance]=[variance_record[variance]]
            variance_record[variance].append(single_order_reader[0])
        variance_list.append(variance)
        tstr = single_order_reader[0]
        single_order_reader[0] = tstr.upper()
        significant_gene_list.append(single_order_reader[0])
    print('After delete genes with p-value>=0.01 or low expression, '+str(len(variance_list))+' genes left.')
    variance_list.sort(reverse=True)
    gene_rank = []
    for single_variance_list in variance_list[0:gene_num]:
        if type(variance_record[single_variance_list]) is str:# 1 variance corresponding to 1 gene
            gene_rank.append(variance_record[single_variance_list])
        else:# 1 variance corresponding to n genes
            gene_rank.append(variance_record[single_variance_list][0])
            del variance_record[single_variance_list][0]
            if len(variance_record[single_variance_list])==1:
                variance_record[single_variance_list]=variance_record[single_variance_list][0]
        if flag:
            f_rank_writer.writerow([variance_record[single_variance_list]])
    f_order.close()
    if flag:
        f_rank.close()
    return gene_rank,significant_gene_list

def get_filtered_gold(gold_network_path,rank_list,output_path,flag):
    #1.Load origin gold file
    #2.Delete genes not in rank_list
    #3.return tf-targets dict and pair-score dict
    #Note: If no score in gold network, score=999
    f_gold = open(gold_network_path,encoding='UTF-8-sig')
    gold_reader = list(csv.reader(f_gold))
    for i in range(0,len(gold_reader)-1):
        temp = gold_reader[i]
        s1 = str(temp[0])
        s2 = str(temp[1])

        temp[0] = s1.upper()
        temp[1] = s2.upper()

        gold_reader[i] = temp
    # print("gold_reader",gold_reader)
    # print("rank_list",rank_list)
    # print("gold_reader",gold_reader)
    print("gold_reader[0]",gold_reader[0])
    has_score=True
    if len(gold_reader[0])<3:
        has_score = False
    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list=[]
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score
        if (single_gold_reader[0] not in rank_list) or (single_gold_reader[1] not in rank_list):
            continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]

        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])
        if str_gene_pair in gold_score_record:
            print('Gold pair repeat!')
        if has_score:
            print("single_gold_reader[2]",single_gold_reader[2])
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 999
        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])
    print("gold_pair_record", gold_pair_record)
    #Some statistics of gold_network
    print(str(len(gold_pair_record)) + ' TFs and ' + str(
            len(gold_score_record)) + ' edges in gold_network consisted of genes in rank_list.')
    print(str(len(unique_gene_list))+' genes are common in rank_list and gold_network.')


    rank_density = len(gold_score_record) / (len(gold_pair_record) * (len(rank_list)))
    gold_density = len(gold_score_record) / (len(gold_pair_record) * (len(unique_gene_list)))

    print('Rank genes density = edges/(TFs*(len(rank_gene)-1))='+str(rank_density))
    print('Gold genes density = edges/(TFs*len(unique_gene_list))=' + str(gold_density))

    #write to file
    print("unique_gene_list",unique_gene_list)
    if flag:
        f_unique = open(output_path, 'w',encoding="utf-8",newline='\n')
        f_unique_writer = csv.writer(f_unique)
        out_unique=np.array(unique_gene_list).reshape(len(unique_gene_list),1)
        f_unique_writer.writerows(out_unique)
        f_unique.close()
    return gold_pair_record,gold_score_record,unique_gene_list

def generate_filtered_gold(gold_pair_record,gold_score_record,output_path):
    # write filtered_gold to output_path
    # print("cnm")
    f_filtered = open(output_path, 'w',encoding="utf-8", newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target', 'Score'])
    # print("cnm")
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            single_output = [tf, target, gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()

def get_gene_pair_list(unique_gene_list, gold_pair_record, gold_score_record, output_file):
    # positive is relationship that tf regulate target
    # negtive is reationship that same tf doesn's regulate target.
    # When same tf doesn't have enough negtive, borrow negtive from other TFs.
    # When negtive is not enough,stop and prove positive:negtive = 1:1

    # generate all negtive gene pairs of TFs
    all_tf_negtive_record = {}
    for tf in gold_pair_record:
        # print("tf",tf)
        all_tf_negtive_record[tf] = []
        for target in unique_gene_list:
            if target in gold_pair_record[tf]:
                continue
            all_tf_negtive_record[tf].append(target)

    # generate negtive record without borrow
    rank_negtive_record = {}
    for tf in gold_pair_record:
        num_positive = len(gold_pair_record[tf])
        if num_positive > len(all_tf_negtive_record[tf]):
            rank_negtive_record[tf] = all_tf_negtive_record[tf]
            all_tf_negtive_record[tf] = []
        else:
            #maybe random.sample(all_tf_negtive_record[tf],num_positive) to promote performance
            rank_negtive_record[tf] = all_tf_negtive_record[tf][:num_positive]
            all_tf_negtive_record[tf] = all_tf_negtive_record[tf][num_positive:]

    # output positive and negtive pairs
    f_gpl = open(output_file, 'w', newline='\n')
    f_gpl_writer = csv.writer(f_gpl)
    f_gpl_writer.writerow(['TF', 'Target', 'Label', 'Score'])
    stop_flag=False
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            # output positive
            single_output = [tf, target, '1', gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
            # output negtive
            if len(rank_negtive_record[tf]) == 0:
                # borrow negtive for other TFs
                find_negtive = False
                for borrow_tf in all_tf_negtive_record:
                    if len(all_tf_negtive_record[borrow_tf]) > 0:
                        find_negtive=True
                        single_output = [borrow_tf, all_tf_negtive_record[borrow_tf][0], 0, 0]
                        del all_tf_negtive_record[borrow_tf][0]
                        break
                # if not enough negtive of others,stop and prove positive:negtive = 1:1
                if not find_negtive:
                    stop_flag = True
                    break
            else:
                #negtive without borrow
                single_output = [tf, rank_negtive_record[tf][0], 0, 0]
                del rank_negtive_record[tf][0]
            once_output.append(single_output)
        if stop_flag:
            f_gpl_writer.writerows(once_output[:-1])
            print('Negtive not enough!')
            break
        f_gpl_writer.writerows(once_output)  # output positive and negtive of 1 TF at a time
    f_gpl.close()


def get_low_express_gene(origin_expression_record,num_cells):
    #get gene_list who were expressed in fewer than 10% of the cells
    gene_list=[]
    threshold=num_cells//10
    for gene in origin_expression_record:
        num=0
        for expression in origin_expression_record[gene]:
            if expression !=0:
                num+=1
                if num>threshold:
                    break
        if num<=threshold:
            gene_list.append(gene)
    return gene_list



def sf(DataSize,index,label):
    new_index = np.arange(DataSize)
    # print("all_index",all_index)
    np.random.shuffle(new_index)
    ntrain_index = []
    nlabel_data = []
    for i in new_index:
        ntrain_index.append(index[i])
        nlabel_data.append(label[i])
    return ntrain_index,nlabel_data

def loadsplit(gold_networks, Rank_nums, names, speciess, results):
    name = names
    Rank_num = Rank_nums
    species = speciess
    result = results
    gold_network = gold_networks
    # ExpressionDataOrdered
    result_dir = species + "/"
    miss = 0
    inp = result_dir
    geneOrdering = name + '_GeneOrdering.csv'
    # gene_expression_path='D:\PyCharmCode2\Code\DGRNS-main' \
    #                      '\mycode20220206\Dataset\gene_expression\\'+gene_Expression

    gene_expression_path = inp + name + '_ExpressionDataOrdered.csv'

    gene_order_path = inp + geneOrdering
    gold_network_path = inp + gold_network + ".csv"

    # output
    save_dir = inp + result + "/"

    datasetName = name + "/"
    # ------------------------------------------------------

    path = save_dir + str(Rank_num) + "/" + gold_network + "/" + datasetName

    Rank_file_name = 'rank.csv'

    rank_path = path + "step1/"

    FGN_file_name = 'FilteredGoldNetwork.csv'
    filtered_path = path + "step1/"

    GPL_file_name = 'GenePairList.csv'
    genePairList_path = path + "step1/"
    # --------------------------------------------------------------------------------

    origin_expression_record, cells = get_origin_expression_data(gene_expression_path)

    Expression_gene_num = len(origin_expression_record)
    Expression_cell_num = len(cells)

    low_express_gene_list = get_low_express_gene(origin_expression_record, len(cells))

    print(str(len(low_express_gene_list)) + ' genes in low expression.')

    for gene in low_express_gene_list:
        origin_expression_record.pop(gene)

    if not os.path.isdir(rank_path):
        os.makedirs(rank_path)

    rank_list, significant_gene_list = \
        get_gene_ranking(gene_order_path, low_express_gene_list, Rank_num, rank_path + Rank_file_name, False)

    for i in range(0, len(rank_list) - 1):
        tstr = str(rank_list[i])
        tstr = tstr.upper()
        rank_list[i] = tstr
        # print("rank_list[i]",rank_list[i])

    # print("rank_list",rank_list)
    print("len(rank_list)", len(rank_list))
    # print("significant_gene_list",significant_gene_list)

    gold_pair_record, gold_score_record, unique_gene_list = \
        get_filtered_gold(gold_network_path, rank_list, rank_path + Rank_file_name, True)
    # print("gold_pair_record",gold_pair_record)
    #
    # print("len(unique_gene_list)",len(unique_gene_list))

    # If origin gold file, generate filtered gold file
    if not os.path.isdir(filtered_path):
        os.makedirs(filtered_path)

    generate_filtered_gold(gold_pair_record, gold_score_record, filtered_path + FGN_file_name)

    # generate gene pair list
    if not os.path.isdir(genePairList_path):
        os.makedirs(genePairList_path)
    get_gene_pair_list(unique_gene_list, gold_pair_record, gold_score_record, genePairList_path + GPL_file_name)

    gene_pair_list_path = result_dir + result + "/" + str(
        Rank_num) + "/" + gold_network + '/' + name + '/step1/GenePairList.csv'

    ###output D:\PyCharmCode2\operation\20220311transform\result
    resultPath = result_dir + result + '/' + str(Rank_num) + "/" + gold_network + "/" + name + "/"

    if (not os.path.isdir(resultPath)):
        os.makedirs(resultPath)

    # Load gene expression data
    origin_expression_record, cells = get_normalized_expression_data(gene_expression_path)
    print("len(origin_expression_record)", len(origin_expression_record))

    # Load gold_pair_record
    all_gene_list = []
    gold_pair_record = {}
    f_genePairList = open(gene_pair_list_path, encoding='UTF-8')  ### read the gene pair and label file

    for single_pair in list(csv.reader(f_genePairList))[1:]:
        if single_pair[2] == '1':
            if single_pair[0] not in gold_pair_record:
                gold_pair_record[single_pair[0]] = [single_pair[1]]
            else:
                gold_pair_record[single_pair[0]].append(single_pair[1])
            # count all genes in gold edges
            if single_pair[0] not in all_gene_list:
                all_gene_list.append(single_pair[0])
            if single_pair[1] not in all_gene_list:
                all_gene_list.append(single_pair[1])
    f_genePairList.close()
    # print dataset statistics
    print('All genes:' + str(len(all_gene_list)))
    print('TFs:' + str(len(gold_pair_record.keys())))
    print("len(single_pair)", len(single_pair))
    # Generate Pearson matrix
    label_list = []
    pair_list = []
    total_matrix = []
    num_tf = -1
    num_label1 = 0
    num_label0 = 0

    x = []
    for i in gold_pair_record:
        num_tf += 1
        for j in range(len(all_gene_list)):
            # for j in range(2):
            print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
            tf_name = i
            target_name = all_gene_list[j]

            flag = False
            if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
                flag = True

            if (flag):
                if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                    label = 1
                    num_label1 += 1
                else:
                    label = 0
                    num_label0 += 1
                label_list.append(label)
                pair_list.append(tf_name + ',' + target_name)

                tf_data = origin_expression_record[tf_name]
                target_data = origin_expression_record[target_name]
            else:
                miss = miss + 1
                continue


            H_T = np.histogram2d(tf_data, target_data, bins=32)
            HT = H_T[0].T

            ##CNNC's input generation
            # a = len(origin_expression_record)
            # print("len(origin_expression_record)", len(origin_expression_record))
            # HT = (log10(HT / a + 10 ** -4) + 4) / 4

            x.append(HT)

            if (len(x) > 0):
                xx = array(x)[:, :, :, newaxis]
            else:
                xx = array(x)

    np.save(resultPath + 'matrix.npy', xx)
    np.save(resultPath + 'label.npy', label_list)
    np.save(resultPath + 'gene_pair.npy', pair_list)

    print('PCC matrix generation finish.')
    print('Positive edges:' + str(num_label1))
    print('Negative edges:' + str(num_label0))
    print('Density=' + str(num_label1 / (num_label1 + num_label0)))

    def split_index(all_index):
        random.shuffle(all_index)
        part = len(all_index) // 5
        train_index = all_index[:3 * part]
        val_index = all_index[3 * part:4 * part]
        test_index = all_index[4 * part:]
        return train_index, val_index, test_index


    data_path = inp + result + "/" + str(Rank_num) + "/" + gold_network + "/" + name + "/"

    label_data = np.load(data_path + 'label.npy')
    num_pairs = len(label_data)

    pos_index = [index for index, value in enumerate(label_data) if value == 1]
    neg_index = [index for index, value in enumerate(label_data) if value == 0]

    pos_train_index, pos_val_index, pos_test_index = split_index(pos_index)

    neg_train_index, neg_val_index, neg_test_index = split_index(neg_index)

    train_index = pos_train_index + neg_train_index
    val_index = pos_val_index + neg_val_index
    test_index = pos_test_index + neg_test_index

    train_label = label_data[train_index]
    val_label = label_data[val_index]
    test_label = label_data[test_index]

    newTrain_index, newTrain_label = sf(len(train_index), train_index, train_label)
    newVal_index, newVal_label = sf(len(val_index), val_index, val_label)
    newTest_index, newTest_label = sf(len(test_index), test_index, test_label)

    with open(data_path + '/train_index.txt', 'w', newline='') as f_train:
        csv_w = csv.writer(f_train, delimiter='\n')
        csv_w.writerow(newTrain_index)

    with open(data_path + '/train_label.txt', 'w', newline='') as f_train:
        csv_w = csv.writer(f_train, delimiter='\n')
        csv_w.writerow(newTrain_label)

    with open(data_path + '/val_index.txt', 'w', newline='') as f_val:
        csv_w = csv.writer(f_val, delimiter='\n')
        csv_w.writerow(newVal_index)

    with open(data_path + '/val_label.txt', 'w', newline='') as f_train:
        csv_w = csv.writer(f_train, delimiter='\n')
        csv_w.writerow(newVal_label)

    with open(data_path + '/test_index.txt', 'w', newline='') as f_test:
        csv_w = csv.writer(f_test, delimiter='\n')
        csv_w.writerow(newTest_index)

    with open(data_path + '/test_label.txt', 'w', newline='') as f_train:
        csv_w = csv.writer(f_train, delimiter='\n')
        csv_w.writerow(newTest_label)

    with open(data_path + '/log.txt', 'w', newline='') as f:
        f.writelines("miss:" + str(miss) + "\n")
        f.writelines('Positive edges:' + str(num_label1) + "\n")
        f.writelines('Negative edges:' + str(num_label0) + "\n")
        f.writelines('Density=' + str(num_label1 / (num_label1 + num_label0)) + "\n")
        f.writelines('All genes:' + str(len(all_gene_list)) + "\n")
        f.writelines('TFs:' + str(len(gold_pair_record.keys())) + "\n")
        f.writelines("len(single_pair)" + str(len(single_pair)) + "\n")
        f.writelines("len(cells)" + str(len(cells)) + "\n")

speciess="mouse"
import time
results = "CNNC"
for j in ['Non-Specific-ChIP-seq-network','STRING-network','mESC-ChIP-seq-network']:
    for k in [500,1000]:
        for i in ['01GM','02L','03E','04mESC','05mDC']:
            f = open("CNNC_time2.txt",mode="a")
            name = j + ":"+str(k) + ":"+i
            start_time = time.time()
            loadsplit(j,k,i,speciess,results)
            end_time = time.time()
            all_time = end_time - start_time
            f.writelines(name+":"+str(all_time)+"\n")
            f.close()
