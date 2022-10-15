# Multi-Scored Sleep Databases: How to Exploit the Multiple-Labels in Automated Sleep Scoring

In this repository we provide an easy-to-use Google Colab Notebook to evaluate DeepSleepNet-Lite 
[[Fiorillo et al.]](https://ieeexplore.ieee.org/abstract/document/9570807) and 
SimpleSleepNet [[Guillot et al.]](https://ieeexplore.ieee.org/abstract/document/9146268) architectures,
as described in our [[arXiv-preprint]](https://arxiv.org/abs/2207.01910). We evaluate the pre-trained models on three 
open access datasets [DOD-H](https://dreem-dod-h.s3.eu-west-3.amazonaws.com/index.html), [DOD-O](https://dreem-dod-o.s3.eu-west-3.amazonaws.com/index.html), [IS-RC](https://stanfordmedicine.app.box.com/s/r9e92ygq0erf7hn5re6j51aaggf50jly/folder/53209541138). Specifically, for each dataset, we upload one of the _k-fold_
pre-trained model. 

[[arXiv-preprint]](https://arxiv.org/abs/2207.01910) Fiorillo L, Pedroncelli D, Favaro P, Faraci FD. Multi-Scored Sleep Databases: How to Exploit the Multiple-Labels 
in Automated Sleep Scoring. arXiv preprint arXiv:2207.01910. 2022 Jul 5. 



Developed by:
Pedroncelli Davide, Fiorillo Luigi

# Requirements
To run our code, you need:
1) [Google Account](https://www.google.com/account)
2) [Google Drive](https://www.google.com/drive)
3) [Google Colab](https://colab.research.google.com)

## Usage

1) __Download data required__ [(link)](https://www.googleapis.com/drive/v3/files/1iSTsJ3BFXDPA6EJXfGoDS9rdhx7Y65Ns?alt=media&key=AIzaSyCqFyfTOhzv5UofENeRHrt7QMGITabRAjA) 
2) __Unzip the file and upload the folder "Experiments" to your Google Drive__ <br />
(_**note**_ do not modify the name/content of the downloaded data, any change could affect the correct execution of the notebook)
3) __Open our Google Colab Notebook__ [(link)](https://colab.research.google.com/drive/1cFwdgoUImooxIBFrZz-_SqkQWCQMD85g?usp=sharing)            
(_**note**_ To speed up the execution, a connection to a GPU runtime is recommended)

Now you are ready to go and to use our Notebook! 

The first step is to run the code cell/block:

```ruby
# Clone git, install libraries
!git clone https://github.com/biomedical-signal-processing/multi-scored-sleep
!pip install torchcontrib

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```
Mounting your Drive is required to access previously uploaded data.

Then, you can run three code cells/blocks:

```ruby
1) !python /content/multi-scored-sleep/SSN/predict.py "DODO" "LSSC"
```

```ruby
2) !python /content/multi-scored-sleep/SSN/predict.py "DODO" "LSSC"
```


These two code blocks perform a prediction with DSNL and SSN respectively. <br />
You can specify two parameteres, as to execute the code on different dataset and pre-trained models:
1) Dataset: "DODO", "DODH" or "ISRC"
2) Pre-trained Model: "LSSC", "LSU", "base"

```ruby
3) !python /content/multi-scored-sleep/plots/plot_subj.py "DSNL" "DODO" "LSSC" "1"
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

(_**note**_ Each time you execute the plot_subj.py script, the .png figures previously generated 
will be automatically overwritten)
