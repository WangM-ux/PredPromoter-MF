
import joblib
import numpy as np
from codes.getFeature import getPromoter, getStrengthPromoter
from codes.readfile import get_data

def getResults(input):
    id, seq, arr = get_data(input)
    result = []
    optim_feature1 = getPromoter(seq, arr)
    promoterModel = joblib.load('./model/Promoter_Best_model.m')
    StrengthModel = joblib.load('./model/Strngth_Best_model.m')
    rs1 = promoterModel.predict_proba(optim_feature1)[:, 1]

    rs1 = rs1.round()
    promoterpos = []
    i = 0
    for r in rs1:
        if r == 1:
            promoterpos.append(i)
        i += 1
    arr2 = np.array(arr)[promoterpos]
    seq2 = np.array(seq)[promoterpos]
    optim_feature2 = getStrengthPromoter(seq2,arr2)
    rs2 = StrengthModel.predict_proba(optim_feature2)[:, 1]
    rs2 = rs2.round()
    i = 0
    for r in rs1:
        if r == 0:
            result.append("Non-promoter")
        elif rs2[i] == 0:
            result.append("Weak-promoter")
            i += 1
        else:
            result.append("Strong-promoter")
            i += 1

    return id, result