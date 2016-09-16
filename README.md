# FER+
This is new label for Emotion FER dataset. Each image is tagged by 10 crowd-sourced taggers, which provide better quality ground truth for still image emotion than the original FER label. Having 10 taggers for each image enables us to create an emotion probability distribution per face so that we can learn a probability or multi-label instead of the conventional majority voting, as described in: https://arxiv.org/abs/1608.01041

The new label file is named: fer2013new.csv, it contains the same number of rows as the original fer2013.csv label file with the same order, so that you infer which emotion tag belong to which image. Since we can't host the actual image content, please find the original FER data set here: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

We also provide a simple parsing code in python to demonstrate how to parse the new label and how to convert it to probability distribution (there is multiple way to do it, we show an example). The parsing code is in src/ReadFERPlus.py

The format of the CSV file is as follow: Usage,	neutral,	happiness,	surprise,	sadness,	anger,	disgust,	fear,	contempt,	unknown,	NF. Where "Usage" is the same as the original FER label to differentiate between training set, public test set and private test set. The other columns are the vote count for each emotion with the addition of unknown and NF (Not a Face).

# Citation
If you use the new FER label or the sample code or part of it in your research, please cite the below:

@inproceedings{BarsoumICMI2016,  
  title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},  
  author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},  
  booktitle={ICMI},  
  year={2016}  
}
