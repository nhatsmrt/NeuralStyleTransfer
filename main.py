from Source import StyleTransferNet
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2
from pathlib import Path

IMG_SIZE = 128

# DEFINE PATHS:
d = Path().resolve()
data_path = str(d) + "/Data/"
predictions_path = str(d) + "/Predictions/"
style_path = data_path + "mouse.png"
train_path = data_path + "/" + str(IMG_SIZE) + "/"
weight_save_path = str(d) + "/weights/model.ckpt"
weight_load_path = str(d) + "/weights/model.ckpt"
pretrained_path = str(d) + "/imagenet-vgg-verydeep-19.mat"

style_img = cv2.imread(style_path).astype(np.float32)
# print(style_img[0])

X = []
names = []
model = StyleTransferNet(
    style_img,
    pretrained_path = pretrained_path
)

for filename in os.listdir(train_path):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue
    img = cv2.imread(train_path + filename) / 255
    names.append(filename)
    X.append(img)

X = np.array(X)
names = np.array(names)
print(X.shape[0])
X_train, X_test, names_train, names_test = train_test_split(X, names, test_size = 0.05)

# mean = np.mean(X_train, axis = 0)
# std = np.mean(X_train, axis = 0)
#
# X_train = (X_train - mean) / std
# X_test = (X_test - mean) / std

model.fit(X_train, weight_save_path = weight_save_path)
model.load_weights(weight_load_path)
predictions = (model.predict(X_test) * 255).astype(np.uint8)

for ind in range(X_test.shape[0]):
    filename = predictions_path + names_test[ind][:-4] + "_styled.png"
    cv2.imwrite(filename, predictions[ind])
