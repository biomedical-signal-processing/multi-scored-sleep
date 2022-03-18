import numpy as np
import os

def cos_sim(p,q):
    return np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))

def load_hypno(path):
    hypno = []
    for fold_idx in range(10):
        f1 = np.load(f"{path}/output_fold{fold_idx}.npz", allow_pickle=True)
        prob_pred = f1["prob_pred"]
        hypno.extend(prob_pred)
    return hypno
def compute_similarity(hypno_sc, hypno):
    cos = []
    cos_distr = []
    std = []
    for i in range(70):
        cos_ = []
        for n, epoch in enumerate(hypno[i]):  # for perchè il mio pc è lento
            cos_.append(cos_sim(epoch, hypno_sc[i][n, :]))
        cos.append(np.mean(cos_))
        std.append(np.std(cos_))
        cos_distr.append(cos_)
    return cos, cos_distr, std

def hypnodensity(L, path, hypno_sc):

    cos_l = []
    std_l = []
    for a in L:
        p = []
        hypno = []
        for fold_idx in range(10):
            f1 = np.load(f"{path}/{a}/output_fold{fold_idx}.npz",
                allow_pickle=True)
            prob_pred = f1["prob_pred"]
            p.extend(prob_pred)
            hypno.append(p)
        for ii in range(len(L)):
            coss = []
            for i in range(55):
                cos = []
                for n, epoch in enumerate(hypno[ii][i]):  # for perchè il mio pc è lento
                    cos.append(cos_sim(epoch, hypno_sc[i][n, :]))
                coss.append(np.mean(cos))
        cos_l.append(np.mean(coss))
        std_l.append(np.std(coss))

    return cos_l, std_l

if __name__ == '__main__':
    # Compute cos similarity across all alpha smoothed:
    base_path = "/content/drive/MyDrive/Experiment _Paper/DOD-O_V2/"    
    files = sorted(os.listdir(base_path))
    test = ['Opsg_0_.npz', 'Opsg_7_.npz', 'Opsg_13_.npz', 'Opsg_20_.npz', 'Opsg_27_.npz','Opsg_54_.npz']
    files = list(set(files) - set(test))
    files.sort(key=lambda x: int(x.split("_")[1]))
    files.extend(test)
    hypno_sc = []
    hypno_lse = []
    hypno_lsu = []
    hypno_base = []
    for file in files:
        f2 = np.load(base_path+file)
        hypno = f2["y_smoothed"]
        hypno_sc.append(hypno[1:len(hypno)-1,:])

    #compute COSINE-SIMILARITY
    base_path_LSE = "/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/prediction/LSE"
    base_path_LSU = "/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/prediction/LSU"
    base_path_base = "/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/prediction/base"
    L = [0]
    cos_base, std_base = hypnodensity(L, base_path_base, hypno_sc)
    L = [0.1, 0.2, 0.3, 0.4, 0.5]
    cos_lsu, std_lsu = hypnodensity(L, base_path_LSU, hypno_sc)
    L = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cos_lse, std_lse = hypnodensity(L, base_path_LSE, hypno_sc)

    print(f"LSE:\n ACS:{np.round(cos_lse,3)}\n ±\n std:{np.round(std_lse,3)}\n")
    print(f"LSU:\n ACS:{np.round(cos_lsu,3)}\n ±\n std:{np.round(std_lsu,3)}\n")
    print(f"base:\n ACS:{np.round(cos_base,3)}\n ±\n std:{np.round(std_base,3)}\n")







































