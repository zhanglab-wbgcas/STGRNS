from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
import os
import csv
import math
from torch.utils.data import (DataLoader)

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_default_tensor_type(torch.DoubleTensor)

def load_index(file):
    with open(file, 'r') as f:
        csv_r = list(csv.reader(f, delimiter='\n'))
    return np.array(csv_r).flatten().astype(int)

def numpy2loader(X, y, batch_size):
    X_set = torch.from_numpy(X)
    X_loader = DataLoader(X_set, batch_size=batch_size)
    y_set = torch.from_numpy(y)
    y_loader = DataLoader(y_set, batch_size=batch_size)

    return X_loader, y_loader

def loaderToList(data_loader):
    length = len(data_loader)
    data = []
    for i in data_loader:
        data.append(i)
    return data

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class STGRNS(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(input_dim, d_model)
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, window_size):
        out = window_size.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        stats = out.mean(dim=1)
        out = self.pred_layer(stats)
        return out

def load_data_TF2(indel_list,data_path):  # cell type specific  ## random samples for reactome is not enough, need borrow some from keggp
    import random
    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:  # len(h_tf_sc)):
        xdata = np.load(data_path + '/Nxdata_tf' + str(i) + '.npy')
        ydata = np.load(data_path + '/ydata_tf' + str(i) + '.npy')
        for k in range(int(len(ydata) / 3)):
            xxdata_list.append(xdata[3 * k, :,
                               :])  ## actually the TF-candidate list we provide has three labels, 1 for TF->target, 2 for target->TF, 0 for TF->non target
            xxdata_list.append(xdata[3 * k + 2, :,
                               :])  ## label 1 0 are selected for interaction task; label 1 2 are selected for causality task.
            yydata.append(1)
            yydata.append(0)
        count_setx = count_setx + int(len(ydata) * 2 / 3)
        count_set.append(count_setx)
        print(i, len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print(np.array(xxdata_list).shape)
    return ((np.array(xxdata_list), yydata_x, count_set))

def STGRNSForGRNSRconstruction(gold_network, name, Rank_num, species, batch_sizes, epochs):
    data_path = 'Dataset/input/' + species + "/" + str(Rank_num) + "/" + gold_network + "/" + name + "/"
    d_models = 200
    # torch.set_num_threads(36) #设置cpu核数
    batch_size = batch_sizes
    log_dir = "log/" + species + "/" + str(Rank_num) + "/" + gold_network + "/" + name + "/"
    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir)

    matrix_data = np.load(data_path + 'matrix.npy')
    label_data = np.load(data_path + 'label.npy')

    x_train, x_t, y_train, y_t = train_test_split(matrix_data, label_data, test_size=0.4, random_state=3,
                                                  stratify=label_data)
    x_val, x_test, y_val, y_test = train_test_split(x_t, y_t, test_size=0.5, random_state=4, stratify=y_t)

    X_trainloader, y_trainloader = numpy2loader(x_train, y_train, batch_size)
    X_valloader, y_valloader = numpy2loader(x_val, y_val, batch_size)
    X_testloader, y_testloader = numpy2loader(x_test, y_test, batch_size)

    X_trainList = loaderToList(X_trainloader)
    y_trainList = loaderToList(y_trainloader)

    X_valList = loaderToList(X_valloader)
    y_valList = loaderToList(y_valloader)

    X_testList = loaderToList(X_testloader)
    y_testList = loaderToList(y_testloader)

    model = STGRNS(input_dim=200, nhead=2, d_model=d_models, num_classes=2)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

    n_epochs = epochs
    acc_record = {'train': [], 'dev': []}
    loss_record = {'train': [], 'dev': []}

    for epoch in range(n_epochs):
        model.train()

        train_loss = []
        train_accs = []

        for j in range(0, len(X_trainList)):
            data = X_trainList[j]
            labels = y_trainList[j]
            logits = model(data)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        acc_record['train'].append(train_acc)
        loss_record['train'].append(train_loss)

        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()
        predictions = []
        labelss = []
        y_test = []
        y_predict = []
        valid_loss = []
        valid_accs = []
        for k in range(0, len(X_valList)):
            data = X_valList[k]
            labels = y_valList[k]
            labels = torch.tensor(labels, dtype=torch.long)
            with torch.no_grad():
                logits = model(data)
            loss = criterion(logits, labels)
            valid_loss.append(loss.item())

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_accs.append(acc)

            predt = F.softmax(logits)
            labelss.extend(labels.cpu().numpy().tolist())
            y_test.extend(labels.cpu().numpy())

            temps = predt.cpu().numpy().tolist()
            for i in temps:
                t = i[1]
                y_predict.append(t)
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        acc_record['dev'].append(valid_acc)
        loss_record['dev'].append(valid_loss)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
        AUPR = metrics.auc(recall, precision)
        acc = metrics.accuracy_score(labelss, predictions)
        bacc = metrics.balanced_accuracy_score(labelss, predictions)
        f1 = metrics.f1_score(labelss, predictions)

        print("acc:", acc, "auc:", auc, "aupr:", AUPR, "bacc", bacc, "f1", f1)

    model_name = str(Rank_num) + "_" + gold_network + "_" + name

    y_test = []
    y_predict = []
    model.eval()
    for k in range(0, len(X_testList)):
        data = X_testList[k]
        labels = y_testList[k]

        with torch.no_grad():
            logits = model(data)
        predt = F.softmax(logits)
        labelss.extend(labels.cpu().numpy().tolist())
        y_test.extend(labels.cpu().numpy())

        # temps = logits.cpu().numpy().tolist()
        temps = predt.cpu().numpy().tolist()
        for i in temps:
            t = i[1]
            y_predict.append(t)
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
    AUPR = metrics.auc(recall, precision)
    acc = metrics.accuracy_score(labelss, predictions)
    bacc = metrics.balanced_accuracy_score(labelss, predictions)
    f1 = metrics.f1_score(labelss, predictions)
    print("acc:", acc, "auc:", auc, "aupr:", AUPR, "bacc", bacc, "f1", f1)


def STGRNSForTF_GenePrediction(name, batch_sizes, epochs, length_TF, th):
    dir = 'Dataset/' + name
    d_models = 200
    input_dim = 200
    torch.set_num_threads(th)
    batch_size = batch_sizes
    for test_indel in range(1, 4):  ################## three fold cross validation
        ## for  3 fold CV
        log_dir = "log/relation_nopro/" + name + "n_epochs=" + str(epochs) + "/" + str(test_indel) + "/"

        if (not os.path.isdir(log_dir)):
            os.makedirs(log_dir)

        whole_data_TF = [i for i in range(length_TF)]

        test_TF = [i for i in range(int(np.ceil((test_indel - 1) * 0.333333 * length_TF)),
                                    int(np.ceil(test_indel * 0.333333 * length_TF)))]  #
        print("test_TF", test_TF)
        train_TF = [i for i in whole_data_TF if i not in test_TF]
        train_TF = np.asarray(train_TF)
        print("len(train_TF)", len(train_TF))

        (x_train, y_train, count_set_train) = load_data_TF2(train_TF, dir)

        from sklearn.model_selection import train_test_split
        seed = 3
        np.random.seed(seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)

        print("x_val.shape", x_val.shape)
        print("nx_train.shape", x_train.shape)

        (x_test, y_test, count_set) = load_data_TF2(test_TF, dir)

        X_trainloader, y_trainloader = numpy2loader(x_train, y_train, batch_size)
        X_valloader, y_valloader = numpy2loader(x_val, y_val, batch_size)
        X_testloader, y_testloader = numpy2loader(x_test, y_test, batch_size)

        X_trainList = loaderToList(X_trainloader)
        y_trainList = loaderToList(y_trainloader)

        X_valList = loaderToList(X_valloader)
        y_valList = loaderToList(y_valloader)

        X_testList = loaderToList(X_testloader)
        y_testList = loaderToList(y_testloader)

        model = STGRNS(input_dim=input_dim, nhead=2, d_model=d_models, num_classes=2)

        criterion = nn.CrossEntropyLoss()
        # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

        n_epochs = epochs
        loss_record = {'train': [], 'dev': []}
        acc_record = {'train': [], 'dev': []}
        for epoch in range(n_epochs):
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_accs = []

            # Iterate the training set by batches.
            for j in range(0, len(X_trainList)):
                data = X_trainList[j]
                labels = y_trainList[j]
                logits = model(data)

                labels = torch.tensor(labels, dtype=torch.long)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()
                acc = (logits.argmax(dim=-1) == labels).float().mean()
                train_accs.append(acc)

                train_loss.append(loss.item())

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            loss_record['train'].append(train_loss)
            acc_record['train'].append(train_acc)

            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

            model.eval()
            valid_loss = []
            valid_accs = []
            predictions = []
            labelss = []
            y_test = []
            y_predict = []

            for k in range(0, len(X_valList)):
                data = X_valList[k]
                labels = y_valList[k]
                labels = torch.tensor(labels, dtype=torch.long)
                with torch.no_grad():
                    logits = model(data)
                loss = criterion(logits, labels)
                valid_loss.append(loss.item())

                acc = (logits.argmax(dim=-1) == labels).float().mean()
                valid_accs.append(acc)

                predt = F.softmax(logits)
                labelss.extend(labels.cpu().numpy().tolist())
                y_test.extend(labels.cpu().numpy())

                temps = predt.cpu().numpy().tolist()
                for i in temps:
                    t = i[1]
                    y_predict.append(t)
                predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            loss_record['dev'].append(valid_loss)
            acc_record['dev'].append(valid_acc)

            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
            AUPR = metrics.auc(recall, precision)
            acc = metrics.accuracy_score(labelss, predictions)
            bacc = metrics.balanced_accuracy_score(labelss, predictions)
            f1 = metrics.f1_score(labelss, predictions)

            print("acc:", acc, "auc:", auc, "aupr:", AUPR, "bacc", bacc, "f1", f1)

        model_path = log_dir + name + '.tar'
        #         print("y_test",y_test)
        model.eval()
        y_test = []
        y_predict = []
        for k in range(0, len(X_testList)):
            data = X_testList[k]
            labels = y_testList[k]
            labels = torch.tensor(labels, dtype=torch.long)
            y_test.extend(labels.cpu().numpy())
            with torch.no_grad():
                logits = model(data)
            predt = F.softmax(logits)
            temps = predt.cpu().numpy().tolist()
            for i in temps:
                t = i[1]
                y_predict.append(t)

        fb = open(log_dir + "result.txt", mode="a")
        fb.writelines(str(test_TF) + "\n")

        np.save(log_dir + 'y_test.npy', y_test)
        np.save(log_dir + 'y_predict.npy', y_predict)
        torch.save(model.state_dict(), model_path)

        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print("all_auc", auc)
        fb.writelines("all_auc:" + str(auc) + "\n")




