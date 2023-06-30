import numpy as np
import pandas as pd
import math
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold

seed = 3
n_splits = 5

def calculate_performace(y_pred, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    FP_index = []
    FN_index = []

    for i in range(len(labels)):
        if y_pred[i] > 0.5 and labels[i] == 1:
            TP += 1
        elif y_pred[i] > 0.5 and labels[i] == 0:
             FP += 1
             FP_index.append(i)
        elif y_pred[i] < 0.5 and labels[i] == 1:
             FN += 1
             FN_index.append(i)
        elif y_pred[i] < 0.5 and labels[i] == 0:
             TN += 1
    Sn = TP / (TP + FN)
    Sp = TN / (FP + TN)
    Acc = (TP + TN) / (TP + FP + TN + FN)
    Mcc = (TP * TN - FP * FN) / math.sqrt((TN + FN) * (FP + TN) * (TP + FN) * (TP + FP))
    precision = TP / (TP + FP)
    f1_score = (2 * precision * Sn) / (precision + Sn)
    roc_auc = metrics.roc_auc_score(labels, y_pred)
    return TP, TN, FP, FN, Sn, Sp, Acc, Mcc, precision, f1_score,  roc_auc


def search_best_parameter(positive, negative, seed, n_splits):

    X = np.vstack(positive, negative)

    X = X.reshape(X.shape[0], X.shape[1]).astype('float32')
    # print(X.shape)
    y = np.array([1] * len(positive) + [0] * len(negative), dtype='float32')

    cv_parameter = {'max_depth': [20, 25, 30, 35],
                'learning_rate': [0.05, 0.1, 0.15, 0.2],
                'n_estimators': [25, 30, 35, 40],
                'gamma': [5, 10, 15, 20],
                'subsample': [0.4, 0.6, 0.8, 1.0],
                'colsample_bytree': [0.5, 0.7, 0.9, 1.0]}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    xgbmodel = XGBClassifier(min_child_weight=1, reg_alpha=0.01, reg_lambda=0, objective='binary:logistic', random_state=3)
    clf = GridSearchCV(estimator=xgbmodel, param_grid=cv_parameter, scoring='accuracy', cv=cv, verbose=0)
    gsearch = clf.fit(X, y)

    return gsearch.best_params_, gsearch.best_score_


def train(positive, negative, seed, n_splits):
    X = np.vstack(positive, negative)
    y = np.array([1] * len(positive) + [0] * len(negative), dtype='float32')


    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    TP = []
    TN = []
    FP = []
    FN = []
    Sn = []
    Sp = []
    Acc = []
    Mcc = []
    precision = []
    f1_score = []
    roc_auc = []


    for train_index, val_index in cv.split(X, y):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        xgbmodel = XGBClassifier(gamma=5, learning_rate=0.15, max_depth=20, n_estimators=40,
                    subsample=0.6, colsample_bytree=0.9, reg_alpha=0.01, min_child_weight=1,
                    reg_lambda=0, objective='binary:logistic', random_state=3)   # 建立模型

        clf = xgbmodel.fit(x_train, y_train)
        y_predict = clf.predict_proba(x_val)
        y_predict = y_predict[:, 1]

        TP1, TN1, FP1, FN1, Sn1, Sp1, Acc1, Mcc1, precision1, f1_score1, roc_auc1 = calculate_performace(y_predict, y_val)
        TP.append(TP1)
        TN.append(TN1)
        FP.append(FP1)
        FN.append(FN1)
        Sn.append(Sn1)
        Sp.append(Sp1)
        Acc.append(Acc1)
        Mcc.append(Mcc1)
        precision.append(precision1)
        f1_score.append(f1_score1)
        roc_auc.append(roc_auc1)


    meanTP = np.mean(TP)
    meanTN = np.mean(TN)
    meanFP = np.mean(FP)
    meanFN = np.mean(FN)

    meanSN = np.mean(Sn)
    meanSP = np.mean(Sp)
    meanACC = np.mean(Acc)
    meanMCC = np.mean(Mcc)
    meanPrecision = np.mean(precision)
    meanF1_score = np.mean(f1_score)
    meanroc_auc = np.mean(roc_auc)

    return meanTP, meanTN, meanFP, meanFN, meanSN, meanSP, meanACC, meanMCC, meanPrecision, meanF1_score, meanroc_auc


def p_datalord(species):
    p_code_file1 = 'data/%s/positive_RCKmer.csv' % species
    p1 = pd.read_csv(p_code_file1, header=None, index_col=0)
    p_code_file2 = 'data/%s/positive_mismatch.csv' % species
    p2 = pd.read_csv(p_code_file2, header=None, index_col=0)
    p_code_file3 = 'data/%s/positive_CKSNAP.csv' % species
    p3 = pd.read_csv(p_code_file3, header=None, index_col=0)
    p_code_file4 = 'data/%s/positive_PseEIIP.csv' % species
    p4 = pd.read_csv(p_code_file4, header=None, index_col=0)
    p_code_file5 = 'data/%s/positive_PseKNC.csv' % species
    p5 = pd.read_csv(p_code_file5, header=None, index_col=0)
    p_data = np.concatenate((p1, p2, p3, p4, p5), axis=1)
    return p_data

def n_datalord(species, data):
    n_code_file1 = 'data/%s/negative_%s_RCKmer.csv' %(species, data)
    n1 = pd.read_csv(n_code_file1, header=None, index_col=0)
    n_code_file2 = 'data/%s/negative_%s_mismatch.csv' %(species, data)
    n2 = pd.read_csv(n_code_file2, header=None, index_col=0)
    n_code_file3 = 'data/%s/negative_%s_CKSNAP.csv' %(species, data)
    n3 = pd.read_csv(n_code_file3, header=None, index_col=0)
    n_code_file4 = 'data/%s/negative_%s_PseEIIP.csv' %(species, data)
    n4 = pd.read_csv(n_code_file4, header=None, index_col=0)
    n_code_file5 = 'data/%s/negative_%s_PseKNC.csv' %(species, data)
    n5 = pd.read_csv(n_code_file5, header=None, index_col=0)
    n_data = np.concatenate((n1, n2, n3, n4, n5), axis=1)
    return n_data

def dataconcat(p_data, n_data):
    X = np.vstack((p_data, n_data))
    X = X.reshape(X.shape[0], 786).astype('float32')
    y = np.array([1] * len(p_data) + [0] * len(n_data), dtype='float32')
    return X, y


def mm_voting(species1, mm_data1, mm_data2, mm_data3, mm_data4, mm_data_test, seed, n_splits):

    xgbmodel = XGBClassifier(gamma=5, learning_rate=0.15, max_depth=20, n_estimators=40,
                             subsample=0.6, colsample_bytree=0.9, reg_alpha=0.01, min_child_weight=1,
                             reg_lambda=0, objective='binary:logistic', random_state=3)

    positive_data = p_datalord(species1)
    negative_data_1 = n_datalord(species1, mm_data1)
    negative_data_2 = n_datalord(species1, mm_data2)
    negative_data_3 = n_datalord(species1, mm_data3)
    negative_data_4 = n_datalord(species1, mm_data4)

    X1, y1 = dataconcat(positive_data, negative_data_1)
    X2, y2 = dataconcat(positive_data, negative_data_2)
    X3, y3 = dataconcat(positive_data, negative_data_3)
    X4, y4 = dataconcat(positive_data, negative_data_4)

    negative_data_test = n_datalord(species1, mm_data_test)
    X_test, y_test = dataconcat(positive_data, negative_data_test)

    TP = []
    TN = []
    FP = []
    FN = []
    Sn = []
    Sp = []
    Acc = []
    Mcc = []
    precision = []
    f1_score = []
    roc_auc = []

    seed = seed
    n_splits = n_splits

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_index, val_index in cv.split(X1, y1):
        x1_train, x1_test = X1[train_index], X_test[val_index]
        y1_train, y1_test = y1[train_index], y_test[val_index]

        x2_train, x2_test = X2[train_index], X_test[val_index]
        y2_train, y2_test = y2[train_index], y_test[val_index]

        x3_train, x3_test = X3[train_index], X_test[val_index]
        y3_train, y3_test = y3[train_index], y_test[val_index]

        x4_train, x4_test = X4[train_index], X_test[val_index]
        y4_train, y4_test = y4[train_index], y_test[val_index]

        # clf1
        clf1 = xgbmodel.fit(x1_train, y1_train)
        val_prediction1 = clf1.predict_proba(x1_test)
        val_prediction1 = val_prediction1[:, 1]
        val_prediction1 = val_prediction1.reshape(val_prediction1.shape[0], 1)

        # clf2
        clf2 = xgbmodel.fit(x2_train, y2_train)
        val_prediction2 = clf2.predict_proba(x2_test)
        val_prediction2 = val_prediction2[:, 1]
        val_prediction2 = val_prediction2.reshape(val_prediction2.shape[0], 1)

        # clf3
        clf3 = xgbmodel.fit(x3_train, y3_train)
        val_prediction3 = clf3.predict_proba(x3_test)
        val_prediction3 = val_prediction3[:, 1]
        val_prediction3 = val_prediction3.reshape(val_prediction3.shape[0], 1)

        # clf4
        clf4 = xgbmodel.fit(x4_train, y4_train)
        val_prediction4 = clf4.predict_proba(x4_test)
        val_prediction4 = val_prediction4[:, 1]
        val_prediction4 = val_prediction4.reshape(val_prediction4.shape[0], 1)

        x_val = np.concatenate((val_prediction1, val_prediction2, val_prediction3, val_prediction4), axis=1)
        print(x_val.shape)
        y_pred = []

        for prob in x_val:
            ave_prob = (prob[0] + prob[1] + prob[2] + prob[3]) / 4
            y_pred.append(ave_prob)
        y_val = y1_test

        TP1, TN1, FP1, FN1, Sn1, Sp1, Acc1, Mcc1, precision1, f1_score1, roc_auc1 = calculate_performace(y_pred, y_val)

        TP.append(TP1)
        TN.append(TN1)
        FP.append(FP1)
        FN.append(FN1)
        Sn.append(Sn1)
        Sp.append(Sp1)
        Acc.append(Acc1)
        Mcc.append(Mcc1)
        precision.append(precision1)
        f1_score.append(f1_score1)
        roc_auc.append(roc_auc1)

    meanTP = np.mean(TP)
    meanTN = np.mean(TN)
    meanFP = np.mean(FP)
    meanFN = np.mean(FN)

    meanSN = np.mean(Sn)
    meanSP = np.mean(Sp)
    meanACC = np.mean(Acc)
    meanMCC = np.mean(Mcc)
    meanPrecision = np.mean(precision)
    meanF1_score = np.mean(f1_score)
    meanroc_auc = np.mean(roc_auc)

    return meanTP, meanTN, meanFP, meanFN, meanSN, meanSP, meanACC, meanMCC, meanPrecision, meanF1_score, meanroc_auc

def hg_voting(species2, hg_data1, hg_data2, hg_data3, hg_data_test, seed, n_splits):

    xgbmodel = XGBClassifier(gamma=5, learning_rate=0.15, max_depth=20, n_estimators=40,
                             subsample=0.6, colsample_bytree=0.9, reg_alpha=0.01, min_child_weight=1,
                             reg_lambda=0, objective='binary:logistic', random_state=3)

    positive_data = p_datalord(species2)
    negative_data_1 = n_datalord(species2, hg_data1)
    negative_data_2 = n_datalord(species2, hg_data2)
    negative_data_3 = n_datalord(species2, hg_data3)


    X1, y1 = dataconcat(positive_data, negative_data_1)
    X2, y2 = dataconcat(positive_data, negative_data_2)
    X3, y3 = dataconcat(positive_data, negative_data_3)


    negative_data_test = n_datalord(species2, hg_data_test)
    X_test, y_test = dataconcat(positive_data, negative_data_test)

    TP = []
    TN = []
    FP = []
    FN = []
    Sn = []
    Sp = []
    Acc = []
    Mcc = []
    precision = []
    f1_score = []
    roc_auc = []

    seed = seed
    n_splits = n_splits

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_index, val_index in cv.split(X1, y1):
        x1_train, x1_test = X1[train_index], X_test[val_index]
        y1_train, y1_test = y1[train_index], y_test[val_index]

        x2_train, x2_test = X2[train_index], X_test[val_index]
        y2_train, y2_test = y2[train_index], y_test[val_index]

        x3_train, x3_test = X3[train_index], X_test[val_index]
        y3_train, y3_test = y3[train_index], y_test[val_index]


        # clf1
        clf1 = xgbmodel.fit(x1_train, y1_train)
        val_prediction1 = clf1.predict_proba(x1_test)
        val_prediction1 = val_prediction1[:, 1]
        val_prediction1 = val_prediction1.reshape(val_prediction1.shape[0], 1)

        # clf2
        clf2 = xgbmodel.fit(x2_train, y2_train)
        val_prediction2 = clf2.predict_proba(x2_test)
        val_prediction2 = val_prediction2[:, 1]
        val_prediction2 = val_prediction2.reshape(val_prediction2.shape[0], 1)

        # clf3
        clf3 = xgbmodel.fit(x3_train, y3_train)
        val_prediction3 = clf3.predict_proba(x3_test)
        val_prediction3 = val_prediction3[:, 1]
        val_prediction3 = val_prediction3.reshape(val_prediction3.shape[0], 1)


        x_val = np.concatenate((val_prediction1, val_prediction2, val_prediction3), axis=1)
        print(x_val.shape)
        y_pred = []

        for prob in x_val:
            ave_prob = (prob[0] + prob[1] + prob[2]) / 3
            y_pred.append(ave_prob)
        y_val = y1_test

        TP1, TN1, FP1, FN1, Sn1, Sp1, Acc1, Mcc1, precision1, f1_score1, roc_auc1 = calculate_performace(y_pred, y_val)

        TP.append(TP1)
        TN.append(TN1)
        FP.append(FP1)
        FN.append(FN1)
        Sn.append(Sn1)
        Sp.append(Sp1)
        Acc.append(Acc1)
        Mcc.append(Mcc1)
        precision.append(precision1)
        f1_score.append(f1_score1)
        roc_auc.append(roc_auc1)

    meanTP = np.mean(TP)
    meanTN = np.mean(TN)
    meanFP = np.mean(FP)
    meanFN = np.mean(FN)

    meanSN = np.mean(Sn)
    meanSP = np.mean(Sp)
    meanACC = np.mean(Acc)
    meanMCC = np.mean(Mcc)
    meanPrecision = np.mean(precision)
    meanF1_score = np.mean(f1_score)
    meanroc_auc = np.mean(roc_auc)

    return meanTP, meanTN, meanFP, meanFN, meanSN, meanSP, meanACC, meanMCC, meanPrecision, meanF1_score, meanroc_auc


if __name__ == '__main__':
    mm_voting(species1='mouse', mm_data1='data1', mm_data2='data2', mm_data3='data3', mm_data4='data4',
              mm_data_test='test', seed=42, n_splits=5)
    hg_voting(species2='human', hg_data1='data1', hg_data2='data2', hg_data3='data3',  hg_data_test='test',
              seed=42, n_splits=5)


