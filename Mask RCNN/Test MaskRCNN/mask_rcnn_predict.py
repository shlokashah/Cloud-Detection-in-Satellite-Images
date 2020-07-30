from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
	help="path to Mask R-CNN model weights")
ap.add_argument("-l", "--labels", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--image", required=True,
	help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())

CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
	NAME = "coco_inference"
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = len(CLASS_NAMES)

config = SimpleConfig()

print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)


image = cv2.imread(args["image"])
# img = cv2.imread(args["image"],0)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)

print("[INFO] making predictions with Mask R-CNN...")
r = model.detect([image], verbose=1)[0]

cloud_coordinates = {}
for i in range(0, r["rois"].shape[0]):
	classID = r["class_ids"][i]
	mask = r["masks"][:, :, i]
	color = COLORS[classID][::-1]
	coordinates = np.where(mask == True)
	x = coordinates[1]
	y = coordinates[0]
	cloud_coordinates[str(i)] = {'x':x,'y':y}
	image = visualize.apply_mask(image, mask, color, alpha=0.5)
	plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
	print(cloud_coordinates)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

count = 0
for i in range(0, len(r["scores"])):
	(startY, startX, endY, endX) = r["rois"][i]
	classID = r["class_ids"][i]
	label = CLASS_NAMES[classID]
	score = r["scores"][i]
	color = [int(c) for c in np.array(COLORS[classID]) * 255]
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	text = "{}:{:}: {:.3f}".format(label,count, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,0.6, color,2)
	count = count + 1

com = defaultdict(list)
for i in range(r['rois'].shape[0]):
	l = (r['rois'][i].tolist())
	x = (l[1]+l[3])/2
	y = (l[0]+l[2])/2
	com[i].append(x)
	com[i].append(y)
# print(r['rois'])
print(r['masks'])
print(com)
with open("com.txt","a") as f:
	print(com,file=f)
plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
