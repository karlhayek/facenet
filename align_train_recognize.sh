##!/bin/bash

# align dataset
python src/align_dataset_mtcnn.py datasets/custom_dataset/raw datasets/custom_dataset/aligned

# train classifier on dataset
python src/classifier.py --mode TRAIN  --data_dir datasets/custom_dataset/aligned/  --model models/20180408-102900/20180408-102900.pb --classifier_filename src/classifier_faces.pkl

## test classifier
#python src/classifier.py --mode CLASSIFY  --data_dir datasets/custom_dataset/aligned/  --model models/20180408-102900/20180408-102900.pb --classifier_filename src/classifier_faces.pkl

## predict
#python src/predict.py --image_files datasets/custom_dataset/test/*/* --model models/20180408-102900/20180408-102900.pb --classifier_filename src/classifier.pkl

# Run real time face recognition with the webcam
python src/real_time_face_recognition.py
