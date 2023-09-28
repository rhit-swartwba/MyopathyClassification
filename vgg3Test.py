from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib as plt
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from os import listdir

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(200, 200))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 200, 200, 3)
    # center pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

predictedValue = []
actualValue = []

# load an image and predict the class
def run_example():
    # load the image
    global predictedValue
    global actualValue
    output = 2
    src_directory = '/Users/blaiseswartwood/Downloads/Dataprep2/Spectrogram/final'
    for file in listdir(src_directory):
        filename = src_directory + '/' + file
        if filename == src_directory + '/.DS_Store':
            continue
        if file.startswith('N'):
            output = 1
        elif file.startswith('M'):
            output = 0
        actualValue.append(output)
        img = load_image(filename)
        # load model
        model = load_model('final_model3.h5')
        # predict the class
        result = model.predict(img)
        result.tolist()
        result = [1 if x > 0.5 else 0 for x in result]
        x = result.pop(0)
        predictedValue.append(x)

    print(actualValue)
    print(predictedValue)


# entry point, run the example
run_example()

    #create confusion matrix and stats
cm = confusion_matrix(actualValue, predictedValue)
print('Confusion Matrix :')
print(cm)
print('Accuracy Score :', accuracy_score(actualValue, predictedValue))
print('Report : ')
print(classification_report(actualValue, predictedValue))

#create confusion matrix table (code taken directly from scikit learn)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#matrix without normalization
plt.figure()
cm_plot_labels=['Myopathy', 'Normal']
plot_confusion_matrix(cm, cm_plot_labels, title="Confusion Matrix")

#matrix with normalization (in % of part)
plt.figure()
plot_confusion_matrix(cm, cm_plot_labels, normalize=True,
                      title='Normalized confusion matrix')
plt.show()





