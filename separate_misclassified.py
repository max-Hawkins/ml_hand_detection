#with a saved model create folders of the misclassified data
#
#follows the below tree
#
# misclassified
#	shouldBeClass1
#		thoughtClass2
#		thoughtClass3
#		...
#	shouldBeClass2
#		...
#	...
#	
#	​
import numpy as np
import argparse, os, shutil
from shutil import copyfile, rmtree
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("-m", help="relative file path to model to be used", type=str)
parser.add_argument("-f", help="relative file path to folder containing data", type=str)
parser.add_argument("-t", help="relative file path for file to be saved default = wd", default='', type=str)
parser.add_argument("-d", help="True will replace the misclassified file instead of writing a new one", default = False, type=bool)
args = parser.parse_args()

#set info from tacs
images_folder = Path(os.getcwd() + "/" + args.f)
model_loc = Path(os.getcwd() + "/" + args.m) 
if args.t =='':
	save_to = Path(os.getcwd())
else:
	save_to = Path(os.getcwd() + "/" + args.t)
	
#get images, labels, classes
test_images = []
test_labels = []
class_names = np.array([item.name for item in images_folder.glob('*')])
for c in class_names:
	temp = os.listdir(images_folder / c)
	for j in temp:
		test_images.append(j)
	for i in range(len(temp)):
		test_labels.append(c)

#import model
model = keras.models.load_model(model_loc)

#create folder to save to (called m)
#either delete the old one and create a new one or add one more depending on -d parameter
m = "misclassified"
if(args.d):
	#delete any folder called misclassified
	for name in os.listdir(save_to): 
		if (name == "misclassified" and os.path.isdir(save_to / name)):
			#FIXME: make sure rmtree is correct
			shutil.rmtree(save_to/"misclassified")
			print("Deleting ", name)
			print("Still need to check")
			exit()
else:
	for i in range(1000):
		if i != 0:
			m = "misclassified"+str(i)
		if not os.path.exists(save_to/m):
			break
		
#get predictions from model
idg = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
batches = idg.flow_from_directory(images_folder,target_size=(300,300),shuffle=False)
predictions = model.predict_generator(batches)
top_guess = [class_names[np.argmax(p)] for p in predictions]

#create folder tree
os.mkdir(save_to/m)
for c in class_names:
	tempS = "shouldBe_"+str(c)
	os.mkdir(save_to/m/tempS)
	for c2 in class_names: 
		if c2 != c:
			tempS2 = "thought_"+str(c2)
			os.mkdir(save_to/m/tempS/tempS2)
			
#find incorrect guesses and save into correct folder
for i in range(len(top_guess)): 
	if top_guess[i]!=test_labels[i]:
		tempS = "shouldBe_"+str(test_labels[i])
		tempS2 = "thought_"+str(top_guess[i])
		copyfile(images_folder / test_labels[i] / test_images[i], save_to/m/tempS/tempS2  / test_images[i])
		
#provide info to user
print("\nIn Dir: ", m)