# -*- coding: utf8 -*-
__author__ = 'yangsongying'
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import os
import csv
import numpy as np
import pandas as pd


def getEmbedding(infile_path, char2id_file):
    char2id, id2char = loadMap(char2id_file)
    row_index = 0
    emb_ext = []
    inc_ext = len(char2id.keys())
    assert(inc_ext not in id2char and inc_ext-1 in id2char)
    with open(infile_path, "r") as infile:
        for row in infile:
            row = row.strip().decode("utf-8")
            row_index += 1
            if row_index == 1:
                num_chars = int(row.split()[0])
                emb_dim = int(row.split()[1])
                emb_matrix = np.zeros((len(char2id.keys()), emb_dim))
                continue
            items = row.split()
            char = items[0]
            emb_vec = [float(val) for val in items[1:]]
            if char in char2id:
                emb_matrix[char2id[char]] = emb_vec
            else:
                id2char[inc_ext] = char
                emb_ext.append(emb_vec)
                inc_ext += 1
            #emb_matrix
        emb_matrix = np.vstack((emb_matrix, np.array(emb_ext)))
    with open(char2id_file, "w") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx) + "\r\n")
    return emb_matrix


def nextBatch(X, y, start_index, batch_size=128):
    last_index = start_index + batch_size
    X_batch = list(X[start_index:min(last_index, len(X))])
    y_batch = list(y[start_index:min(last_index, len(X))])
    if last_index > len(X):
        left_size = last_index - (len(X))
        for i in range(left_size):
            index = np.random.randint(len(X))
            X_batch.append(X[index])
            y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


def nextRandomBatch(X, y, batch_size=128):
    X_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(X))
        X_batch.append(X[index])
        y_batch.append(y[index])
    X_batch = np.array(X_batch)
    y_batch = np.array(y_batch)
    return X_batch, y_batch


# use "0" to padding the sentence
def padding(sample, seq_max_len):
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def prepare(chars, labels, seq_max_len, is_padding=True):
    X = []
    y = []
    tmp_x = []
    tmp_y = []

    for c, l in zip(chars, labels):
        # empty line
        if c == -1:
            if len(tmp_x) <= seq_max_len:
                X.append(tmp_x)
                y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c)
            tmp_y.append(l)

    if (len(tmp_x) > 0 and len(tmp_x) <= seq_max_len):
        X.append(tmp_x)
        y.append(tmp_y)

    if is_padding: #补全长度
        X = np.array(padding(X, seq_max_len))
        y = np.array(padding(y, seq_max_len))
    else:
        X = np.array(X)
        y = np.array(y)

    return X, y


def getTrain(train_path, val_path=None, train_val_ratio=0.95,
             char2id_file='char2id', label2id_file='label2id',
             seq_max_len=50):
    char2id, id2char, label2id, id2label = buildMap(train_path, char2id_file, label2id_file)

    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                           names=["char", "label"])

    # map the char and label into id
    df_train["char_id"] = df_train.char.map(lambda x: -1 if str(x) == str(np.nan) else char2id[x])
    df_train["label_id"] = df_train.label.map(lambda x: -1 if str(x) == str(np.nan) else label2id[x])

    # convert the data in maxtrix
    X, y = prepare(df_train["char_id"], df_train["label_id"], seq_max_len)
    num_samples = len(X)

    # shuffle the samples
    # indexs = np.arange(num_samples)
    # np.random.shuffle(indexs)
    # X = X[indexs]
    # y = y[indexs]

    if val_path != None:
        X_train = X
        y_train = y
        X_val, y_val, _, _ = getTrain(val_path, train_val_ratio=1.0, seq_max_len=seq_max_len)

    elif train_val_ratio >= 1.0:
        X_train = X
        y_train = y
        X_val = None
        y_val = None
    else:
        # split the data into train and validation set
        X_train = X[:int(num_samples * train_val_ratio)]
        y_train = y[:int(num_samples * train_val_ratio)]
        X_val = X[int(num_samples * train_val_ratio):]
        y_val = y[int(num_samples * train_val_ratio):]

    print "train size: %d, validation size: %d" % (len(X_train), len(y_val))

    return X_train, y_train, X_val, y_val

def getTest(test_path, seq_max_len=200, char2id_file='char2id'):
    char2id, id2char = loadMap(char2id_file)

    df_test = pd.read_csv(test_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                          names=["char", "label"])

    def mapFunc(x, char2id):
        if str(x) == str(np.nan):
            return -1
        xd = x.decode("utf-8")
        if xd not in char2id:
            return char2id["<NEW>"]
        else:
            return char2id[xd]

    df_test["char_id"] = df_test.char.map(lambda x: mapFunc(x, char2id))
    # df_test["char"] = df_test.char.map(lambda x: -1 if str(x) == str(np.nan) else x)
    X_test, X_test_Txt = prepare(df_test["char_id"], df_test.char, seq_max_len)
    return X_test, X_test_Txt


def genCharId(sentence, char2id, seq_max_len=200):
    length = len(sentence)
    if length > seq_max_len:
        length = seq_max_len
    charid = []
    for i in range(length):
        c = sentence[i]
        charid.append(char2id[c] if c in char2id else char2id["<NEW>"])
    if length < seq_max_len:
        charid += [0 for _ in range(seq_max_len - length)]
    return np.array(charid).reshape(1, seq_max_len)


def genLabelId(labels, label2id):
    res = []
    for l in labels:
        res.append(label2id[l] if l in label2id else 0)
    return res


def loadMap(token2id_filepath):
    token2id = {}
    id2token = {}
    with open(token2id_filepath) as infile:
        for row in infile:
            row = row.rstrip().decode("utf-8")
            token = row.split('\t')[0]
            token_id = int(row.split('\t')[1])
            token2id[token] = token_id
            id2token[token_id] = token
    return token2id, id2token


def saveMap(id2char, id2label, char2id_file, label2id_file):
    with open(char2id_file, "wb") as outfile:
        for idx in id2char:
            outfile.write(id2char[idx] + "\t" + str(idx) + "\r\n")
    with open(label2id_file, "wb") as outfile:
        for idx in id2label:
            outfile.write(str(id2label[idx]) + "\t" + str(idx) + "\r\n")
    print "saved map between token and id"


def buildMap(train_path="train.in", char2id_file='char2id', label2id_file='label2id'):
    df_train = pd.read_csv(train_path, delimiter='\t', quoting=csv.QUOTE_NONE, skip_blank_lines=False, header=None,
                           names=["char", "label"])
    chars = list( set( df_train["char"][ df_train["char"].notnull() ] ) )
    labels = list(set( df_train["label"][ df_train["label"].notnull() ] ) )
    char2id = dict(zip(chars, range(1, len(chars) + 1)))
    label2id = dict(zip(labels, range(1, len(labels) + 1)))
    id2char = dict(zip(range(1, len(chars) + 1), chars))
    id2label = dict(zip(range(1, len(labels) + 1), labels))
    id2char[0] = "<PAD>"
    char2id["<PAD>"] = 0

    id2char[len(chars) + 1] = "<NEW>"
    char2id["<NEW>"] = len(chars) + 1

    label2id["<PAD>"] = 0
    id2label[0] = "<PAD>"
    saveMap(id2char, id2label, char2id_file, label2id_file)

    return char2id, id2char, label2id, id2label


def parseTag(tags, line, id2label):
    res = []
    tmp = []
    for i in range(len(tags)):
        tag = id2label[tags[i]]
        bies = tag[0]
        t = tag[2:]
        tmp.append(line[i])
        if bies == "S" or bies == "E":
            res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
            tmp = []

    if len(tmp) > 0:
        res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
    return " ".join(res)


def parseTag_Ner(tags, line, id2label):
    res = []
    tmp = []
    for i in range(len(tags)):
        tag = id2label[tags[i]]
        bies = tag[0]
        t = tag[2:]
        if bies == "B":
            if len(tmp) > 0:
                res.append("".join(tmp))
                tmp = []
        tmp.append(line[i])
        if bies == "S" or bies == "E":
            res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
            tmp = []

    if len(tmp) > 0:
        res.append("".join(tmp) if len(t) == 0 else "".join(tmp) + "/" + t)
    return " ".join(res)
