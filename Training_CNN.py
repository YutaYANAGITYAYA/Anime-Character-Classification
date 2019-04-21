import sys
import os.path as op
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt

TITLE = sys.argv[1]
labels, col = np.loadtxt(op.join(TITLE,'Labels.tsv'), delimiter='\t', dtype='S20,S20', unpack=True)
IMG_SHAPE = (100, 128, 3)


def read_img(fn):
    img = cv2.imread(fn)
    # サイズ調整
    img = cv2.resize(img, (np.max(IMG_SHAPE),np.max(IMG_SHAPE)))
    img = img[:IMG_SHAPE[0],:IMG_SHAPE[1],:]
    # ガウシアンフィルタで平滑化
    img = cv2.GaussianBlur(img, (5,5), 1)
    # 輝度のみ平坦化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img[:,:,0] = cv2.equalizeHist(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img

def mk_data(fn):
    print('Loading...', fn)
    labels, lst = np.loadtxt(fn, delimiter='\t', dtype='i8,S50', unpack=True)
    images = np.stack([read_img(fn.decode()) for fn in lst])
    images = images / 255.0
    return labels, images

try:
    data = np.load(op.join(TITLE,'train.npz'))
    train_labels = data['lbl']
    train_images = data['img']
    data = np.load(op.join(TITLE,'test.npz'))
    test_labels = data['lbl']
    test_images = data['img']
    ans = input('Use save data? [Y/n] ')
    if ans in ['N','n','No','no']:
        raise Exception
except:
    train_labels, train_images = mk_data(op.join(TITLE,'train.tsv'))
    np.savez(op.join(TITLE,'train.npz'),lbl=train_labels,img=train_images)
    test_labels,  test_images  = mk_data(op.join(TITLE,'test.tsv'))
    np.savez(op.join(TITLE,'test.npz'), lbl=test_labels, img=test_images)


# Create the convolutional base
kernel_size = (5, 5)
model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size, activation='relu', input_shape=IMG_SHAPE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size, activation='relu'))
# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))


# Compile and train the model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)


## Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
model.save(op.join(TITLE,'model.h5'))


## Make predictions
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    img = np.array(img * 255, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(labels[predicted_label].decode(),
                                    100*np.max(predictions_array),
                                    labels[true_label].decode()),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(labels)), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

plt.show()