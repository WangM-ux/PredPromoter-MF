
import joblib
import numpy as np
from codes.getFeature import *
from codes.readfile import get_data

def getResults(input):
    id, seq, arr = get_data(input)
    result = []
    feature = getPromoter(seq, arr)
    promoterModel = joblib.load('./model/promoter_Best_model.m')
    sigma70Model = joblib.load('./model/sigma70_Best_model.m')
    sigma24Model = joblib.load('./model/sigma24_Best_model.m')
    sigma32Model = joblib.load('./model/sigma32_Best_model.m')
    sigma38Model = joblib.load('./model/sigma38_Best_model.m')
    sigma28Model = joblib.load('./model/sigma28_Best_model.m')

    promoterFea = getoptimPromoter(feature, 'promoter')
    sigma70Fea = getoptimPromoter(feature, 'sigma70')
    sigma24Fea = getoptimPromoter(feature, 'sigma24')
    sigma32Fea = getoptimPromoter(feature, 'sigma32')
    sigma38Fea = getoptimPromoter(feature, 'sigma38')
    sigma28Fea = getoptimPromoter(feature, 'sigma28')

    rsPromoter = promoterModel.predict_proba(promoterFea)[:, 1].round()
    rsSigma70 = sigma70Model.predict_proba(sigma70Fea)[:, 1].round()
    rsSigma24 = sigma24Model.predict_proba(sigma24Fea)[:, 1].round()
    rsSigma32 = sigma32Model.predict_proba(sigma32Fea)[:, 1].round()
    rsSigma38 = sigma38Model.predict_proba(sigma38Fea)[:, 1].round()
    rsSigma28 = sigma28Model.predict_proba(sigma28Fea)[:, 1].round()

    i = 0
    for r in rsPromoter:
        rstmp = ''
        if r == 0:
            rstmp += 'Non-promoter  '
        elif rsSigma70[i] == 1:
            rstmp += 'Sigma70-promoter  '
        elif rsSigma24[i] == 1:
            rstmp += 'Sigma24-promoter  '
        elif rsSigma32[i] == 1:
            rstmp += 'Sigma32-promoter  '
        elif rsSigma38[i] == 1:
            rstmp += 'Sigma38-promoter  '
        elif rsSigma28[i] == 1:
            rstmp += 'Sigma28-promoter  '
        else:
            rstmp += 'Sigma54-promoter  '

        result.append(rstmp)
        i += 1

    return id, result