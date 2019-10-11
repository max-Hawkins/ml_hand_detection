import numpy as np
import argparse, os
from random import shuffle
from shutil import copyfile
from pathlib import Path

# Defines user inputs
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="percentage of total data to save as training data", default=80, type=int)
parser.add_argument("-f", help="relative file path to folder containing complete data", type=str)
args = parser.parse_args()

train_percent = args.p / 100.0
test_percent = (100 - args.p) / 100.0

# Print out user input data
print("\nTrain: " + str(train_percent))
print("Test: " + str(test_percent))

complete_folder = Path(os.getcwd() + "\\" + args.f)
train_folder = Path(os.getcwd() + "/train")
test_folder = Path(os.getcwd() + "/test")

print("Data Folder: " + str(complete_folder))

# Delete any files in the training and test directories
for file in os.listdir(train_folder):
    if os.path.exists(train_folder / file) and os.path.isfile(train_folder / file):
      os.remove(train_folder / file)
    else:
      print(file + " does not exist")

for file in os.listdir(test_folder):
    if os.path.exists(test_folder / file) and os.path.isfile(test_folder / file):
      os.remove(test_folder / file)
    else:
      print(file + " does not exist")

# Create and shuffle lists of directories containing labeled data
att_files = os.listdir(complete_folder / "Attentive") 
dis_files = os.listdir(complete_folder / "Distracted")
shuffle(att_files)
shuffle(dis_files)

att_cutoff = int(train_percent * len(att_files))
dis_cutoff = int(train_percent * len(dis_files))

train_att_files = att_files[:att_cutoff]
train_dis_files = dis_files[:dis_cutoff]

test_att_files = att_files[att_cutoff:]
test_dis_files = dis_files[dis_cutoff:]

# Transfers subset of original data into training and test folders
for file in train_att_files:
    copyfile(complete_folder / "Attentive" / file, train_folder / 'attentive' / file)

for file in train_dis_files:
    copyfile(complete_folder / "Distracted" / file, train_folder / 'distracted' / file)

for file in test_att_files:
    copyfile(complete_folder / "Attentive" / file, test_folder / 'attentive' / file)

for file in test_dis_files:
    copyfile(complete_folder / "Distracted" / file, test_folder / 'distracted' / file)

print('\n' + str(len(train_att_files) + len(train_dis_files)) + ' Training Files')
print(str(len(test_att_files) + len(test_dis_files)) + ' Test Files')