import sys, os
from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

def cos_sim(p,q):
    return np.dot(p,q)/(np.linalg.norm(p)*np.linalg.norm(q))

def acs(hyp_true, hyp_pred):
    cos_ = []
    for i in range(len(hyp_pred)):
      cos_.append(cos_sim(hyp_pred[i],hyp_true[i]))
    cos = np.mean(cos_)
    return cos

# Retrive settings from cm
arch = sys.argv[1]
dataset = sys.argv[2]
model = sys.argv[3]
idx_paz = int(sys.argv[4])

if dataset == "DODO":
  output = np.load(f"/content/drive/MyDrive/Experiments/plot_data/{arch}/output_fold1_{dataset}_{model}.npz",allow_pickle=True)
  allfiles = ['Opsg_6_.npz', 'Opsg_8_.npz', 'Opsg_9_.npz', 'Opsg_10_.npz', 'Opsg_11_.npz'] 
  data = np.load(f"/content/drive/MyDrive/Experiments/data/DODO/{allfiles[idx_paz]}", allow_pickle=True)

elif dataset == "DODH":
  output = np.load(f"/content/drive/MyDrive/Experiments/plot_data/{arch}/output_fold24_{dataset}_{model}.npz",allow_pickle=True)
  allfiles = ['Hpsg_24_.npz'] 
  data = np.load(f"/content/drive/MyDrive/Experiments/data/DODH/{allfiles[idx_paz]}", allow_pickle=True)

elif dataset == "ISRC":
  output = np.load(f"/content/drive/MyDrive/Experiments/plot_data/{arch}/output_fold4_{dataset}_{model}.npz",allow_pickle=True)
  allfiles = ['AL_29_030107.npz', 'AL_30_061108.npz', 'AL_31_010909.npz', 'AL_32_032408.npz', 'AL_33_042207.npz', 'AL_34_101908.npz', 'AL_35_032607.npz']
  data = np.load(f"/content/drive/MyDrive/Experiments/data/ISRC/{allfiles[idx_paz]}", allow_pickle=True)


y_true = output["y_true"][idx_paz]
hyp_pred = np.asarray(output["prob_pred"][idx_paz])
y_pred = [np.argmax(i) for i in hyp_pred]
hyp_true = data["y_smoothed"]
hyp_true = hyp_true[1:len(hyp_true)-1]

print(f"Architecture: {arch} {model}")
print(f"Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}")

acc = np.round(accuracy_score(y_true, y_pred)*100,1)
print(f"Accurcay : {acc}")
acs_ = np.round(acs(hyp_true,hyp_pred),3)
print(f"ACS : {acs_}")


#Hypnogram True
fig, axes = plt.subplots(2, 1, dpi=300)
fig.set_size_inches(30, 12)
plt.subplot(2, 1, 1)
plt.title(f"Hypnogram - Ground Truth, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}", fontsize = 20)
x = list(range(0, len(y_true)))
# plt.plot(x, y_true,  color="teal")
plt.plot(x, y_true,  color="black")
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Sleep Stages",fontsize=15)
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0,1,2,3,4],["W","N1","N2","N3","R"],fontsize=15)
plt.xlim([0,len(y_true)])

#Hypnogram Pred
plt.subplot(2, 1, 2)
plt.title(f"Hypnogram - Predicted, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}, Accuracy = {acc}%", fontsize = 20)

# plt.plot(x, y_true,  color="teal")
plt.plot(x, y_pred,  color="black")
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Sleep Stages",fontsize=15)
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0,1,2,3,4],["W","N1","N2","N3","R"],fontsize=15)
plt.xlim([0,len(y_true)])

# set the spacing between subplots
plt.subplots_adjust(hspace=0.3)
plt.savefig('/content/Figure_Hypnogram.png',dpi=300)
print("Figure_Hypnogram.png saved to the path /content/Figure_Hypnogram.png")


# Hypnodensity Pred
fig, axes = plt.subplots(2, 1, dpi=300)
fig.set_size_inches(30, 12)
n_epoch = np.shape(hyp_pred)[0]
D = {
    "W": hyp_pred[:,0],
    "N1":hyp_pred[:,1],
    "N2":hyp_pred[:,2],
    "N3":hyp_pred[:,3],
    "REM":hyp_pred[:,4]
}

df = pd.DataFrame(D)
# color = ["mediumpurple","lightsteelblue","cornflowerblue","royalblue","indigo"]
color = ["aliceblue","lightsteelblue","cornflowerblue","royalblue","tab:red"]
df.plot(kind="bar", stacked=True, width=1, color=color, rot=0,ax=axes[1], title=f"Hypnodensity Graph - Predicted, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}, Accuracy = {acc}%, ACS = {acs_}", fontsize = 20)
plt.sca(axes[1])
axes[1].title.set_size(20)
# Per legend non sovrapposta
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=False, ncol=5, fontsize=15)

#plt.legend(loc="upper right")
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0, 0.25, 0.5 ,0.75, 1],["0", 0.25, 0.5 ,0.75, "1"],fontsize=15)
plt.ylim([0,1])
plt.xlim([0,len(y_true)])
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Probability",fontsize=15)

# Hypnodensity True
n_epoch = np.shape(hyp_true)[0]
D = {
    "W": hyp_true[:,0],
    "N1":hyp_true[:,1],
    "N2":hyp_true[:,2],
    "N3":hyp_true[:,3],
    "REM":hyp_true[:,4]
}

df = pd.DataFrame(D)
# color = ["mediumpurple","lightsteelblue","cornflowerblue","royalblue","indigo"]
color = ["aliceblue","lightsteelblue","cornflowerblue","royalblue","tab:red"]
df.plot(kind="bar", stacked=True, width=1, color=color, rot=0,ax=axes[0],legend=None, title=f"Hypnodensity Graph - Ground Truth, Architecture: {arch} {model}, Dataset: {dataset}, Subject: {allfiles[idx_paz][:-5]}", fontsize = 20)
plt.sca(axes[0])
axes[0].title.set_size(20)

# Per legend non sovrapposta
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
#          fancybox=True, shadow=False, ncol=5, fontsize=15)

#plt.legend(loc="upper right")
plt.xticks([0,200,400,600,800,1000,1200,1400],[0,100,200,300,400,500,600,700],fontsize=15)
plt.yticks([0, 0.25, 0.5 ,0.75, 1],["0", 0.25, 0.5 ,0.75, "1"],fontsize=15)
plt.ylim([0,1])
plt.xlim([0,len(y_true)])
plt.xlabel("Time [min]",fontsize=15)
plt.ylabel("Probability",fontsize=15)


# set the spacing between subplots
plt.subplots_adjust(hspace=0.4)

plt.savefig(f'/content/Figure_Hypnodensity.png',dpi=300)
print("Figure_Hypnodensity.png saved to the path /content/Figure_Hypnodensity.png")
