# FER+
The FER+ annotations provide a set of new labels for the standard Emotion FER dataset. In FER+, each image has been labeled by 10 crowd-sourced taggers, which provide better quality ground truth for still image emotion than the original FER labels. Having 10 taggers for each image enables researchers to estimate an emotion probability distribution per face. This allows constructing algorithms that produce statistical distributions or multi-label outputs instead of the conventional single-label output, as described in: https://arxiv.org/abs/1608.01041

Here are some examples of the FER vs FER+ labels extracted from the abovementioned paper (FER top, FER+ bottom):
![FER vs FER+ example](https://raw.githubusercontent.com/Microsoft/FERPlus/master/FER+vsFER.png)

The new label file is named **_fer2013new.csv_** and contains the same number of rows as the original *fer2013.csv* label file with the same order, so that you infer which emotion tag belong to which image. Since we can't host the actual image content, please find the original FER data set here: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

We also provide a simple parsing code in Python to show how to parse the new label and how to convert it to a probability distribution (there are multiple ways to do it, we show an example). The parsing code is in [src/ReadFERPlus.py](https://github.com/Microsoft/FERPlus/tree/master/src)

The format of the CSV file is as follow: usage,	neutral, happiness,	surprise, sadness, anger, disgust, fear, contempt, unknown, NF. Columns "usage" is the same as the original FER label to differentiate between training, public test and private test sets. The other columns are the vote count for each emotion with the addition of unknown and NF (Not a Face).

# Citation
If you use the new FER+ label or the sample code or part of it in your research, please cite the below:

**@inproceedings{BarsoumICMI2016,  
&nbsp;&nbsp;&nbsp;&nbsp;title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},  
&nbsp;&nbsp;&nbsp;&nbsp;booktitle={ICMI},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2016}  
}**
