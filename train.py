import matplotlib
#matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

args = {'dataset': 'data', 'model': 'model/activity_gpu.model', 'label_bin': 'model/lb.pickle', 'epochs': 25, 'plot':'plot.png'}

LABELS = set(["Cyclone", "Earthquake", "Flood", "Wildfire"])

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
	label = imagePath.split(os.path.sep)[-2]
	if label not in LABELS:
		continue
	image = cv2.imread(imagePath)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	data.append(image)
	labels.append(label)

data = np.array(data)
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# 80% train-test split
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# initialize the validation/testing data augmentation object
valAug = ImageDataGenerator()

mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# load the ResNet-50 network, ensuring the head FC layer sets are left off
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)

# place the head FC model on top of the base model - actual trained model
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / args["epochs"])
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("[INFO] training head...")
H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=32),
	steps_per_epoch=len(trainX) // 32,
	validation_data=valAug.flow(testX, testY),
	validation_steps=len(testX) // 32,
	epochs=args["epochs"])

print("[INFO] evaluating network...")
y_score = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	y_score.argmax(axis=1), target_names=lb.classes_))

# Plotting curves
lw = 2
from sklearn.metrics import roc_curve, auc
from scipy import interp
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = model.predict(testX, batch_size=32)
n_classes = 4
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
   
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])    
plt.clf()
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='navy', linestyle=':', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('ROC.eps', format='eps', dpi=1000)
plt.savefig('ROC.png', format='png', dpi=1000)
# PR Curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(testY[:, i],
                                                            y_score[:, i])
    average_precision[i] = average_precision_score(testY[:, i], y_score[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(testY.ravel(),
        y_score.ravel())
average_precision["micro"] = average_precision_score(testY, y_score,
                                                         average="micro")
plt.clf()
plt.figure(2)
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Average precision score over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
plt.savefig('Precision-Recall.eps', format='eps', dpi=1000)
plt.savefig('Precision-Recall.png', format='png', bbox_inches='tight')

'''
#-------------------------------
# plot the training loss and accuracy
N = args["epochs"]
plt.style.use("ggplot")
plt.figure(3)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
'''
#------------------Plot Learning Curve---------------
'''
plt.clf() 
plt.figure(4)
plt.plot(H.history["val_loss"], label="val_loss")
plt.plot(H.history["val_acc"], label="val_acc")
plt.plot(H.history["loss"], label="train_loss")
plt.plot(H.history["acc"], label="train_acc")
plt.title('Model Learning Curve')
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epoch')
plt.legend(['Validation loss', 'Validation accuracy', 'Training loss', 'Training accuracy'], loc='upper right')
plt.savefig('Learning-curve.eps', format='eps', dpi=1000)
plt.savefig('Learning-curve.png', format='png', bbox_inches='tight')
#------------------------------------------Con MAT2---------------============================-----
def plot_confusion_matrix1(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

y_test = testY.argmax(axis=1)
y_pred = y_score.argmax(axis=1)
lb = ["Cyclone", "Earthquake", "Flood", "Wildfire"] #Thunderstorm, Building_Collapse
# Plot normalized confusion matrix
plot_confusion_matrix1(y_test, y_pred, classes=lb, normalize=True,
                      title='Normalized confusion matrix')

#plt.show()
#plt.clf()
#plt.figure(6)
#plt.savefig('CM.eps', format='eps', dpi=1000)
plt.savefig('CM.png', format='png', dpi=1000)
plt.savefig('CM.tiff', format='tiff', dpi=1000)
'''
#-----------------------------------------------------------------------
# serialize the model to disk
print("[INFO] Saving model...")
model.save(args["model"])

# serialize the label binarizer to disk
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()