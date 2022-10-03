# Multi-Scored Sleep Databases: How to Exploit the Multiple-Labels in Automated Sleep Scoring

This repository provides the code and a Google Colab Notebook to test some of the architectures presented in the paper "Multi-Scored Sleep Databases: How to Exploit the Multiple-Labels in Automated Sleep Scoring".
 As an example, some subjects from 3 datasets are available: DODO, DODH and ISRC. Those considered most representative and with a fair % of examples for each class (W, N1, N2, N3, REM) were selected.

Developed by:
Davide Pedroncelli, Lugi Fiorillo
# Requirements
To run our code, you need:
1) Google Account https://www.google.com/account/
2) Google Drive https://www.google.com/drive/
3) Google Colab https://colab.research.google.com/

## Usage

1) __Download data required__ https://drive.google.com/drive/folders/19PWnnIpQQ8cZ4c_Vw-xfQ4MMCoGglbPN?usp=sharing
2) __Unzip the file and upload it to your Gogle Drive__ (N.B. Do not modify the content of the downloaded data, any change could affect the correct execution of the notebook)
3) __Open our Google Colab Notebook__ https://colab.research.google.com/drive/1cFwdgoUImooxIBFrZz-_SqkQWCQMD85g?usp=sharing

Now you are ready to use our Notebook! 
The first step is to run the fisrt code block:

```ruby
# Clone git and install python libraries needed
!git clone https://Davide-Pedroncelli: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@github.com/Davide-Pedroncelli/Notebook_for_SLEEP.git
!pip install torchcontrib
```
Then, you can choose between 3 code blocks:

```ruby
1) !python /content/Notebook_for_SLEEP/DSNL/predict.py "DODH" "LSU"
```

```ruby
2) !python /content/Notebook_for_SLEEP/SSN/predict.py "DODH" "LSSC"
```


These code blocks perform a prediction with DSNL and SSN respectively. 
There are two parameteres:
1) Dataset - "DODO", "DODH", "ISRC"
2) Model - "LSSC", "LSU", "base"

You can modify the parameters to execute the code on different dataset and trained model.

```ruby
3) !python /content/Notebook_for_SLEEP/Plot_Subj.py "SSN" "ISRC" "LSU" "0"
```
This code block generates two figures of a subject:
1) Figure_Hypnogram.png
2) Figure_Hypnodensity.png

There are four parameteres:
1) Architecture - "DSNL", "SSN"
2) Dataset - "DODO", "DODH", "ISRC"
3) Model - "LSSC", "LSU", "base"
4) Subject Index - from 0 to 4 for DODO, 0 for DODH, from 0 to 6 for ISRC (each index is a different subject of the dataset)

As before, you can change the parameters at will.

(N.B. Each time Plot_Subj.py is executed, previous png figures are automatically overwritten)
