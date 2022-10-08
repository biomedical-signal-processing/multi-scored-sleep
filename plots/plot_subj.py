import sys, os
from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))) 
from metrics.ece_acs import *

# Read sys params
arch = sys.argv[1]
dataset = sys.argv[2]
model = sys.argv[3]
idx_paz = int(sys.argv[4])

# Path, dirs
if dataset == "DODO":
  output = np.load(f"/content/drive/MyDrive/Experiments/plot_data/{arch}/output_fold1_{dataset}_{model}.npz",allow_pickle=True)
  allfiles = ['Opsg_6_.npz', 'Opsg_8_.npz', 'Opsg_9_.npz', 'Opsg_10_.npz', 'Opsg_11_.npz'] 
  data = np.load(f"/content/drive/MyDrive/Experiments/data/DODO/{allfiles[idx_paz]}", allow_pickle=True)

elif dataset == "DODH":
  output = np.load(f"/content/drive/MyDrive/Experiments/plot_data/{arch}/output_fold24_{dataset}_{model}.npz",allow_pickle=True)
  allfiles = ['Hpsg_24_.npz']

elif dataset == "ISRC":
  output = np.load(f"/content/drive/MyDrive/Experiments/plot_data/{arch}/output_fold4_{dataset}_{model}.npz",allow_pickle=True)
  allfiles = ['AL_29_030107.npz', 'AL_30_061108.npz', 'AL_31_010909.npz', 'AL_32_032408.npz', 'AL_33_042207.npz', 'AL_34_101908.npz', 'AL_35_032607.npz']

# Loading predicted data
y_true = np.array(output["y_true"][idx_paz])
y_pred = np.array(output["y_pred"][idx_paz])
hypno_true = np.asarray(output["hypno_true"][idx_paz])
hypno_pred = np.asarray(output["hypno_pred"][idx_paz])

# Computing accuracy
acc = np.round(accuracy_score(y_true, y_pred)*100,1)
print(f"Accurcay : {acc}")
# Computing ACS
acs_ = np.round(compute_acs(hypno_true,hypno_pred),3)
print(f"ACS : {acs_}")

# Computing % per class
w = f"W = {round(len(np.where(y_true == 0)[0])/len(y_true)*100,2)}%"
n1 = f"N1 = {round(len(np.where(y_true == 1)[0])/len(y_true)*100,2)}%"
n2 = f"N2 = {round(len(np.where(y_true == 2)[0])/len(y_true)*100,2)}%"
n3 = f"N3 = {round(len(np.where(y_true == 3)[0])/len(y_true)*100,2)}%"
rem = f"REM = {round(len(np.where(y_true == 4)[0])/len(y_true)*100,2)}%"
print(f"% examples: {w}, {n1}, {n2}, {n3}, {rem}")

# Print info
print(f"Architecture: {arch} {model}")
print(f"Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}")

# Plot Hypnogram True
fig, axes = plt.subplots(2, 1, dpi=300)
fig.set_size_inches(30, 12)
plt.subplot(2, 1, 1)
plt.title(f"Hypnogram - Ground Truth, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}", fontsize = 20)
x = list(range(0, len(y_true)))
plt.plot(x, y_true,  color="black")
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Sleep Stages",fontsize=15)
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0,1,2,3,4],["W","N1","N2","N3","R"],fontsize=15)
plt.xlim([0,len(y_true)])

# Plot Hypnogram Pred
plt.subplot(2, 1, 2)
plt.title(f"Hypnogram - Predicted, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}, Accuracy = {acc}%", fontsize = 20)
plt.plot(x, y_pred,  color="black")
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Sleep Stages",fontsize=15)
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0,1,2,3,4],["W","N1","N2","N3","R"],fontsize=15)
plt.xlim([0,len(y_true)])
# set the spacing between subplots
plt.subplots_adjust(hspace=0.3)

# Saving Hypnogram Figure
plt.savefig('/content/Figure_Hypnogram.png',dpi=300)
print("Figure_Hypnogram.png saved to the path /content/Figure_Hypnogram.png")


# Plot Hypnodensity Pred
fig, axes = plt.subplots(2, 1, dpi=300)
fig.set_size_inches(30, 12)
D = {
    "W": hypno_pred[:,0],
    "N1":hypno_pred[:,1],
    "N2":hypno_pred[:,2],
    "N3":hypno_pred[:,3],
    "REM":hypno_pred[:,4]
}

df = pd.DataFrame(D)
color = ["aliceblue","lightsteelblue","cornflowerblue","royalblue","tab:red"]
df.plot(kind="bar", stacked=True, width=1, color=color, rot=0,ax=axes[1], title=f"Hypnodensity Graph - Predicted, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}, Accuracy = {acc}%, ACS = {acs_}", fontsize = 20)
plt.sca(axes[1])
axes[1].title.set_size(20)
# Legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=False, ncol=5, fontsize=15)

plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0, 0.25, 0.5 ,0.75, 1],["0", 0.25, 0.5 ,0.75, "1"],fontsize=15)
plt.ylim([0,1])
plt.xlim([0,len(y_true)])
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Probability",fontsize=15)

# Plot Hypnodensity True
D = {
    "W": hypno_true[:,0],
    "N1":hypno_true[:,1],
    "N2":hypno_true[:,2],
    "N3":hypno_true[:,3],
    "REM":hypno_true[:,4]
}

df = pd.DataFrame(D)
color = ["aliceblue","lightsteelblue","cornflowerblue","royalblue","tab:red"]
df.plot(kind="bar", stacked=True, width=1, color=color, rot=0,ax=axes[0],legend=None, title=f"Hypnodensity Graph - Ground Truth, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}", fontsize = 20)
plt.sca(axes[0])
axes[0].title.set_size(20)
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0, 0.25, 0.5 ,0.75, 1],["0", 0.25, 0.5 ,0.75, "1"],fontsize=15)
plt.ylim([0,1])
plt.xlim([0,len(y_true)])
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Probability",fontsize=15)
# set the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Saving Hypnodensity Figure
plt.savefig(f'/content/Figure_Hypnodensity.png',dpi=300)
print("Figure_Hypnodensity.png saved to the path /content/Figure_Hypnodensity.png")

