# Multi-Scored Sleep Databases: How to Exploit the Multiple-Labels in Automated Sleep Scoring

In this repository we provide an easy-to-use Google Colab Notebook to evaluate DeepSleepNet-Lite 
[[Fiorillo et al.]](https://ieeexplore.ieee.org/abstract/document/9570807) and 
SimpleSleepNet [[Guillot et al.]](https://ieeexplore.ieee.org/abstract/document/9146268) architectures,
as described in our [[arXiv-preprint]](https://arxiv.org/abs/2207.01910). We evaluate the pre-trained models on three 
open access datasets [DOD-H](), [DOD-O](), [IS-RC](). Specifically, for each dataset, we upload one of the _k-fold_
pre-trained model. 

[[arXiv-preprint]](https://arxiv.org/abs/2207.01910) Fiorillo L, Pedroncelli D, Favaro P, Faraci FD. Multi-Scored Sleep Databases: How to Exploit the Multiple-Labels 
in Automated Sleep Scoring. arXiv preprint arXiv:2207.01910. 2022 Jul 5. 



Developed by:
Davide Pedroncelli, Lugi Fiorillo

# Requirements
To run our code, you need:
1) Google Account https://www.google.com/account
2) Google Drive https://www.google.com/drive
3) Google Colab https://colab.research.google.com

## Usage

1) __Download data required__ https://drive.google.com/drive/folders/19PWnnIpQQ8cZ4c_Vw-xfQ4MMCoGglbPN?usp=sharing
2) __Unzip the file and upload it to your Gogle Drive__ <br />
(_**note**_ do not modify the content of the downloaded data, any change could affect the correct execution of the notebook)
3) __Open our Google Colab Notebook__ https://colab.research.google.com/drive/1cFwdgoUImooxIBFrZz-_SqkQWCQMD85g?usp=sharing

Now you are ready to go and to use our Notebook! 

The first step is to run the first code cell/block:

```ruby
# Clone git and install python libraries needed
!git clone https://github.com/bio-signal-processing/multi-scored-sleep
!pip install torchcontrib
```
Then, you can run three code cells/blocks:

```ruby
1) !python /content/multi-scored-sleep/DSNL/predict.py "DODH" "LSU"
```

```ruby
2) !python /content/multi-scored-sleep/SSN/predict.py "DODH" "LSSC"
```


These two code blocks perform a prediction with DSNL and SSN respectively. <br />
You can specify two parameteres, as to execute the code on different dataset and pre-trained models:
1) Dataset: "DODO", "DODH" or "ISRC"
2) Pre-trained Model: "LSSC", "LSU", "base"

```ruby
3) !python /content/multi-scored-sleep/Plot_Subj.py "SSN" "ISRC" "LSU" "0"
```
This code block generates the hypnogram and the hypnodensity-graph for a specific test-subject:
1) Figure_Hypnogram.png
2) Figure_Hypnodensity.png

You can specify four parameteres:
1) Architecture: "DSNL", "SSN"
2) Dataset: "DODO", "DODH", "ISRC"
3) Pre-trained Model: "LSSC", "LSU", "base"
4) Subject Index: from 0 to 4 for DODO, 0 for DODH, from 0 to 6 for ISRC <br /> 
(for each dataset each index correspond to a different test-subject)

(_**note**_ Each time you execute the Plot_Subj.py script, the .png figures previously generated 
will be automatically overwritten)
