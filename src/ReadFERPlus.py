#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import sys
import csv
import argparse
import numpy as np

def main(fer_label_file):
    """
    Main entry points, it simply parse the new FER emotion label file and print its summary.

    Parameters:
    fer_label_file: Path to the CSV label file.
    """
    header, train_data, val_data, test_data = load_labels(fer_label_file)

    # Print the summary using the emotion with max probability (majority voting).
    emotion_count = len(header)
    train_image_count_per_emotion      = count_image_per_emotion(emotion_count, train_data)
    validation_image_count_per_emotion = count_image_per_emotion(emotion_count, val_data)
    test_image_count_per_emotion       = count_image_per_emotion(emotion_count, test_data)

    print("{0}\t{1}\t{2}\t{3}".format("".ljust(10), "Train", "Val", "Test"))

    for index in range(emotion_count): 
        print("{0}\t{1}\t{2}\t{3}".format(header[index].ljust(10), 
                                          train_image_count_per_emotion[index], 
                                          validation_image_count_per_emotion[index], 
                                          test_image_count_per_emotion[index]))

def count_image_per_emotion(emotion_count, data):
    """
    For summary display, a helper function that count the number of
    image per emotion.

    Parameters:
    emotion_count: the number of emotions.
    data: the list of emotion for each image.
    """
    image_count_per_emotion = [0] * emotion_count
    for emotion_prob in data:
        image_count_per_emotion[np.argmax(emotion_prob)] += 1

    return image_count_per_emotion

def load_labels(fer_label_file):
    """
    Load and parse the label CSV file, contains the new FER label.

    Parameters:
    fer_label_file: Path to the CSV label file.
    """    
    train_data = []
    val_data   = []
    test_data  = []

    with open(fer_label_file) as label_file: 
        emotion_label = csv.reader(label_file)
        emotion_label_itr = iter(emotion_label)

        # First row is the header
        header = next(emotion_label_itr)
        header = header[1:len(header)]

        # Split into train, validate and test set.
        for row in emotion_label_itr:
            emotion_raw = map(float, row[1:len(row)])
            if row[0] == "Training":
                train_data.append(process_data(emotion_raw))
            elif row[0] == "PublicTest":
                val_data.append(process_data(emotion_raw))
            elif row[0] == "PrivateTest":
                test_data.append(process_data(emotion_raw))
            else:
                raise ValueError('Invalid usage')

    return header, train_data, val_data, test_data

def process_data(emotion_raw):
    """
    Takes the raw votes for each emotion and return the probability distribution. 
    We ignore outliers and distribution that has one vote per emotion.

    Parameters:
    emotion_raw: Array of vote count per emotion from the label file.
    """
    size = len(emotion_raw) 
    emotion_unknown = [0.0] * size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal) 
    for i in range(size):
        if emotion_raw[i] < 1.0 + sys.float_info.epsilon:
            emotion_raw[i] = 0.0

    sum_list = sum(emotion_raw)
    emotion = [0.0] * size 

    sum_part = 0
    count = 0
    valid_emotion = True
    while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
        maxval = max(emotion_raw) 
        for i in range(size): 
            if emotion_raw[i] == maxval: 
                emotion[i] = maxval
                emotion_raw[i] = 0
                sum_part += emotion[i]
                count += 1
                if i >= 8:  # unknown or non-face share same number of max votes 
                    valid_emotion = False
                    if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                        emotion[i] = 0
                        count -= 1
                    break
    if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
        emotion = emotion_unknown   # force setting as unknown 

    return [float(i)/sum(emotion) for i in emotion]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fer_label_file", type = str, help = "FER 2013 update label file.", required = True)

    args = parser.parse_args()

    main(args.fer_label_file)