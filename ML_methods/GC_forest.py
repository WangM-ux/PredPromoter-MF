#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from ML_methods.gcforest.gcforest import GCForest
import warnings
warnings.filterwarnings("ignore")


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["window"] = 100
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 300, "max_depth": 5,
              "nthread": -1, "learning_rate": 0.1, 'objective': 'binary:logistic', 'nthread': 4,
             'scale_pos_weight': 1, 'seed': 30})
    # ca_config["estimators"].append({"n_folds": 5, "type": "GradientBoostingClassifier", "n_estimators": 10})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config
    # ca_config["estimators"].append({"n_folds": 5, "type": "LGBMClassifier", "n_estimators": 10, "max_depth": 30, "n_jobs": -1})
    # ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})

def GCForest_Classifier(model, X, y, indep=None, fold=10, out='gcforest_output'):
    print('##############################GCForest prediction##############################')
    classes = sorted(list(set(y)))
    ntest = len(indep)
    if ntest != 0:
        oof_test = np.zeros((ntest))
        oof_test_skf = np.empty((fold, ntest))

    prediction_result_cv = []
    acctmp = 0

    folds = StratifiedKFold(fold, shuffle=True).split(X, y)

    for i, (trained, valided) in enumerate(folds):
        train_y, train_X = y[trained], X[trained]
        valid_y, valid_X = y[valided], X[valided]

        if len(valided) == 0:
            continue

        model.fit_transform(train_X, train_y,valid_X,valid_y)
        y_pred = model.predict(valid_X)
        acc = accuracy_score(valid_y, y_pred)
        print("************第" + str(i) + "次交叉验证结果************")
        print("************Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
        if acc > acctmp:
            # 持久化模型
            print("Model save......")
            save_path_name = out + "_Best_model.m"
            joblib.dump(model, save_path_name, compress=3)
        acctmp = acc

        scores = model.predict_proba(valid_X)
        tmp_result = np.zeros((len(valid_y), len(classes) + 1))
        tmp_result[:, 0], tmp_result[:, 1:] = valid_y, scores
        prediction_result_cv.append(tmp_result)

        # independent
        if indep.shape[0] != 0:
            oof_test_skf[i] = model.predict_proba(indep)[:, 1]

    print(oof_test_skf.shape)
    # 对独立测试的五次预测结果求取平均值
    oof_test[:] = oof_test_skf.mean(axis=0)
    print(oof_test.shape)
    print(oof_test.reshape(-1, 1).shape)
    print(oof_test)
    return oof_test.reshape(-1, 1), prediction_result_cv


def gcforest_nonTest(model, X, y, fold=10, out='gcforest_output'):
    print('##############################GCForest prediction##############################')
    classes = sorted(list(set(y)))

    prediction_result_cv = []
    prediction_acc_cv = []
    # indices = np.arange(len(X))
    # np.random.shuffle(indices)
    # folds = StratifiedKFold(fold).split(indices, y)
    folds = StratifiedKFold(fold, shuffle=True).split(X, y)
    for i, (trained, valided) in enumerate(folds):
        train_y, train_X = np.array(y)[trained], np.array(X)[trained]
        valid_y, valid_X = np.array(y)[valided], np.array(X)[valided]

        if len(valided) == 0:
            continue

        rfc = model.fit_transform(train_X, train_y)
        y_pred = model.predict(valid_X)
        acc = accuracy_score(valid_y, y_pred)
        prediction_acc_cv.append(acc)
        print("************第"+str(i)+"次交叉验证结果************")
        print("************Test Accuracy of GcForest = {:.2f} %".format(acc * 100))
        scores = model.predict_proba(valid_X)
        tmp_result = np.zeros((len(valid_y), len(classes) + 1))
        tmp_result[:, 0], tmp_result[:, 1:] = valid_y, scores
        prediction_result_cv.append(tmp_result)

    return prediction_result_cv, prediction_acc_cv, rfc

def gcforest_train(X, Y, featurename,indepX, indepY,modelFile):

    model = GCForest(config=get_toy_config())
    weidu = len(X[0])
    template = "{0},{1},{2},{3},{4},{5},{6},{7}\n"

    print('Sum of samples is ' + str(len(Y)))
    print('Dimension of ' + featurename + ' is ' + str(len(X[0])))
    print('Sum of independent samples is ' + str(len(indepY)))
    print('Dimension of ' + featurename + ' is ' + str(len(indepX[0])))
    cv_indep, cv_train= GCForest_Classifier(model, X, Y, indepX,out=modelFile)