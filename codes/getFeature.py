
from sklearn.preprocessing import StandardScaler

from codes.extra_hidden_features import create_truncated_model, create_trained_model
from codes.extra_kmer_feature import Kmer
from codes.extre_pse_feature import get_pseknc
from codes.readfile import *

# 提取最佳特征组合
def optim_fea_fromindep(x, posfile):
    # 加载最佳特征位置
    posarr = []
    with open(posfile, 'r', encoding='UTF-8') as inputfile:
        while True:
            lines = inputfile.readline()
            if not lines:
                break
                pass
            posarr.append(int(lines))

    x = x[:, posarr]
    return x
kw = {'order': 'ACGT', }
def getPromoter(seq, arr):
    kmer = 5
    Kmerfeature = Kmer(arr, kmer, **kw)
    PseKNCfeature = get_pseknc(arr, kmer=5, lamda=11, w=0.9)

    seqs = seq
    mono_fea = get_Mono_mer(seqs)
    tri_fea = get_Tri_mer(seqs)
    di_sp = get_SP_Di_Nucleotide(seqs)
    tri_sp = get_SP_Tri_nucleotide(seqs)
    # 原模型
    trained_model = create_trained_model(f"./model/best_weight.h5")
    # 这是截取从原模型的输入层到中间层的新模型，并加载原模型参数
    truncated_model = create_truncated_model(trained_model)
    # 提取隐层特征
    hidden_features = truncated_model.predict([mono_fea, tri_fea, di_sp, tri_sp])
    # 输出隐层特征
    df_1 = hidden_features[0]
    df_2 = hidden_features[1]
    df_3 = hidden_features[2]
    df_4 = hidden_features[3]

    fea = np.c_[Kmerfeature, PseKNCfeature, df_1, df_2, df_3, df_4]
    # ZScore标准化
    ss = StandardScaler()
    fea = ss.fit_transform(fea)
    optimfea = optim_fea_fromindep(fea, './dataset/optimpromoter_pos.txt')

    return optimfea

def getStrengthPromoter(seq,arr):
    kmer = 3
    Kmerfeature = Kmer(arr, kmer, **kw)
    PseKNCfeature = get_pseknc(arr, kmer=3, lamda=13, w=0.7)

    seqs = seq
    mono_fea = get_Mono_mer(seqs)
    tri_fea = get_Tri_mer(seqs)
    di_sp = get_SP_Di_Nucleotide(seqs)
    tri_sp = get_SP_Tri_nucleotide(seqs)
    # 原模型
    trained_model = create_trained_model(f"./model/best_weight_strength.h5")
    # 这是截取从原模型的输入层到中间层的新模型，并加载原模型参数
    truncated_model = create_truncated_model(trained_model)
    # 提取隐层特征
    hidden_features = truncated_model.predict([mono_fea, tri_fea, di_sp, tri_sp])
    # 输出隐层特征
    df_1 = hidden_features[0]
    df_2 = hidden_features[1]
    df_3 = hidden_features[2]
    df_4 = hidden_features[3]

    fea = np.c_[Kmerfeature, PseKNCfeature, df_1, df_2, df_3, df_4]
    # ZScore标准化
    ss = StandardScaler()
    fea = ss.fit_transform(fea)
    optimfea = optim_fea_fromindep(fea, './dataset/optimstrength_pos.txt')

    return optimfea
