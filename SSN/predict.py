import numpy as np
from dreem_learning_open.datasets.dataset import DreemDataset
from dreem_learning_open.models.modulo_net.net import ModuloNet
from dreem_learning_open.trainers import Trainer
from dreem_learning_open.preprocessings.h5_to_memmap import h5_to_memmaps
import json
import sys, os
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
import pandas as pd
from datetime import datetime
import time
from tabulate import tabulate
import warnings
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 
from metrics.ece_acs import *

# Disable UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# Read sys params
dataset = sys.argv[1]
model = sys.argv[2]

# Path, dirs
base_path = "/content/drive/MyDrive/Experiments/SSN"
model_dir = f"{base_path}/{dataset}/{model}"
output_dir = "/content/pred"
groups_description = json.load(open(rf"/content/drive/MyDrive/Experiments/data/memmap/{dataset}/groups_description.json"))
features_description = json.load(open(rf"/content/drive/MyDrive/Experiments/data/memmap/{dataset}/features_description.json"))

if dataset == "DODO":
  fold_idx = 1
  test_files = ['Opsg_6_', 'Opsg_8_', 'Opsg_9_', 'Opsg_10_', 'Opsg_11_'] 
elif dataset == "DODH":
  fold_idx = 24
  test_files = ['Hpsg_24_']
elif dataset == "ISRC":
  fold_idx = 4
  test_files = ['AL_29_030107', 'AL_30_061108', 'AL_31_010909', 'AL_32_032408', 'AL_33_042207', 'AL_34_101908', 'AL_35_032607']

# Fold path
folds = [f"{model_dir}/fold{fold_idx}"]

# Test files path
test_files = [f"/content/drive/MyDrive/Experiments/data/memmap/{dataset}/{i}"for i in test_files]

# Useful Vatiables
ece = []
acs = []
acc = []
mf1 = []
wf1 = []
k = []
f1_w = []
f1_n1 = []
f1_n2 = []
f1_n3 = []
f1_r = []
all_y_true = []
all_y_pred = []
all_hypno_true = []
all_hypno_pred = []

print(f"Architecture: SSN, Model: {model}, Estimated time (~ 60 s per subject)\n")

for i, fold in enumerate(folds):

    # Loading test dataset
    dataset_test = DreemDataset(groups_description, features_description=features_description,
                                temporal_context=21,
                                records=test_files)
    # Disable print (to avoid unnecessary information)
    blockPrint()

    # Loading best_net
    net = ModuloNet.load(fr'{fold}/best_model.gz')
    trainer_parameters = {
        "epochs": 100,
        "patience": 15,
        "ls": {
        "type": "uniform",
        "alpha": 0.0
    },
        "optimizer": {
            "type": "adam",
            "args": {
                "lr": 1e-3
            }
        }
    }

    # Init trainer
    trainer = Trainer(net=net,
                      **trainer_parameters)
    # Enable print
    enablePrint()
    
    # Prediction
    start_time = time.time() 
    print(f"\n[{datetime.now()}] Predicting...\n")
    performance_on_test_set, _, performance_per_records, hypnograms = trainer.validate(dataset_test,
                                                                                       return_metrics_per_records=True,
                                                                                       verbose=True)
    for ii, key in enumerate(list(hypnograms.keys())):
        
        # Load data
        y_true = np.array(hypnograms[key]["target"])
        y_pred = np.array(hypnograms[key]["predicted"])
        hypno_pred = np.array(hypnograms[key]["prob"])
        hypno_true = np.load(f"{test_files[ii]}/soft_consensus.npz")["soft_consensus"]
        mask = (np.array([y in [0, 1, 2, 3, 4] for y in y_true]))
        
   
        # Delete unlabeled data (-1)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        hypno_pred = hypno_pred[mask,:]
        hypno_true = hypno_true[mask,:]
        y_conf = []
        for i in range(len(hypno_pred)):
          y_conf.append(np.max(hypno_pred[i,:]))
        
        # Storing all date to save
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_hypno_true.append(hypno_true)
        all_hypno_pred.append(hypno_pred)

        # Compute ECE
        ece.append(compute_ece(np.array(y_true), np.array(y_pred), np.array(y_conf), num_bins=20))

        # Compute ACS
        acs.append(compute_acs(hypno_pred, hypno_true))

        # Compute Performance
        acc.append(np.mean(accuracy_score(y_true, y_pred)))
        mf1.append(f1_score(y_true, y_pred, average="macro"))
        k.append(cohen_kappa_score(y_true, y_pred))
        wf1.append(f1_score(y_true, y_pred, average="weighted"))
        f1_ = f1_score(y_true, y_pred, average=None)
        cls = np.setdiff1d(np.array([0,1,2,3,4]),np.unique(y_pred))
        # Check if there is at least one example for each class
        d = {}
        for n,i in enumerate(np.unique(y_pred)):
          d[i] = f1_[n]
        if cls.size > 0:
          for i in cls:
            d[i] = np.nan
        f1_w.append(d[0])
        f1_n1.append(d[1])
        f1_n2.append(d[2])
        f1_n3.append(d[3])
        f1_r.append(d[4])

duration = np.round(time.time() - start_time,1)
print(f"\nDone! [Time elapsed: {duration} s]")

# Overall performance
Acc = np.round(np.mean(acc)*100,1)
MF1 = np.round(np.mean(mf1)*100,1)
K = np.round(np.mean(k)*100,1)
WF1 = np.round(np.mean(wf1)*100,1)
F1_w = np.round(np.nanmean(f1_w)*100,1)
F1_n1 = np.round(np.nanmean(f1_n1)*100,1)
F1_n2 = np.round(np.nanmean(f1_n2)*100,1)
F1_n3 = np.round(np.nanmean(f1_n3)*100,1)
F1_r = np.round(np.nanmean(f1_r)*100,1)

acc_ = np.round(np.mean(acc),3)
conf = []
for k in ece:
  conf.append(k['avg_confidence'])
conf = np.round(np.mean(conf),3)
ece_ = round(abs(acc_ - conf),3)
acs = f"{np.round(np.mean(acs),3)} ± {np.round(np.std(acs),3)}"

# Print Table Overall Performance
print("\nOverall Performance Tables: \n")
print(tabulate([[dataset, f"SSN {model}", Acc, MF1, WF1, K, F1_w, F1_n1, F1_n2, F1_n3, F1_r]], headers=['Dataset','Model','Accuracy %', 'MF1 %', 'WF1 %','Cohen-k %', 'W %', 'N1 %', 'N2 %','N3 %','REM %'], tablefmt="pretty"))
print(tabulate([[dataset, f"SSN {model}", ece_, acc_, conf, acs]], headers=['Dataset','Model','ECE', 'Accuracy', 'Confidence','ACS'],tablefmt="pretty"))

# Saving Prediction
save_dict = {
  "y_true" : all_y_true,
  "y_pred":all_y_pred,
  "hypno_true":all_hypno_true,
  "hypno_pred":all_hypno_pred
}

np.savez(f"/content/drive/MyDrive/Experiments/plot_data/SSN/output_fold{fold_idx}_{dataset}_{model}.npz", **save_dict)
print(f"\nPrediction saved to path /content/drive/MyDrive/Experiments/plot_data/SSN/output_fold{fold_idx}_{dataset}_{model}.npz")
