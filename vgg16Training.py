from keras.utils.vis_utils import plot_model
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


# define cnn model
def define_model():
   # load model
   model = VGG16(include_top=False, input_shape=(224, 224, 3))
   # mark loaded layers as not trainable
   for layer in model.layers:
       layer.trainable = False
   # connected layers
   flat1 = Flatten()(model.layers[-1].output)
   class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
   drop1 = Dropout(0.2)(class1)
   class2 = Dense(64, activation='relu', kernel_initializer='he_uniform')(drop1)
   drop2 = Dropout(0.2)(class2)
   class3 = Dense(32, activation='relu', kernel_initializer='he_uniform')(drop2)
   class4 = Dense(16, activation='relu', kernel_initializer='he_uniform')(class3)
   output = Dense(1, activation='sigmoid')(class4)
   # define new model
   model = Model(inputs=model.inputs, outputs=output)
   #model summary
   print(model.summary())
   plot_model(model, to_file='tl3_model_plot.png', show_shapes=True, show_layer_names=True)
   # compile model
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model


# plot diagnostic learning curves
def loss_curve(history):
   # plot loss
   pyplot.plot()
   pyplot.title('Cross Entropy Loss')
   pyplot.plot(history.history['loss'], color='blue', label='train')
   pyplot.plot(history.history['val_loss'], color='orange', label='test')
   pyplot.xlabel("Epochs")
   pyplot.ylabel("Loss")
   pyplot.legend(['train', 'test'], loc='upper left')
   # save plot to file
   pyplot.savefig('tl3_model_loss_plot.png')
   pyplot.close()

def accuracy_curve(history):
   # plot accuracy
   pyplot.plot()
   pyplot.title('Classification Accuracy')
   pyplot.plot(history.history['accuracy'], color='blue', label='train')
   pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
   pyplot.xlabel("Epochs")
   pyplot.ylabel("Accuracy")
   pyplot.legend(['train', 'test'], loc='upper left')
   # save plot to file
   pyplot.savefig('tl3_model_accuracy_plot.png')
   pyplot.close()
   loss_curve(history)


# run the test harness for evaluating a model
def run_test_harness():
   # define model
   model = define_model()
   # create data generator
   train_datagen = ImageDataGenerator(featurewise_center=True, horizontal_flip=True)
   test_datagen = ImageDataGenerator(featurewise_center=True)
   # specify imagenet mean values for centering
   train_datagen.mean = [123.68, 116.779, 103.939]
   test_datagen.mean = [123.68, 116.779, 103.939]
   # prepare iterator
   #what change
   train_it = train_datagen.flow_from_directory('/Users/blaiseswartwood/Downloads/Dataprep2/Scalogram/train',
                                          class_mode='binary', batch_size=64, target_size=(224, 224))
   test_it = test_datagen.flow_from_directory('/Users/blaiseswartwood/Downloads/Dataprep2/Scalogram/test',
                                         class_mode='binary', batch_size=64, target_size=(224, 224))
   # fit model
   history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                 validation_data=test_it, validation_steps=len(test_it), shuffle=True, epochs=25, verbose=1)
   # evaluate model
   _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
   print('> %.3f' % (acc * 100.0))
   # learning curves
   accuracy_curve(history)
   #save model
   model.save('tl3_final_model.h5')


# entry point, run the test harness
run_test_harness()
