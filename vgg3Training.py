#import packages
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
   model = Sequential()
   #3 block VGG
   model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                    input_shape=(200, 200, 3)))
   model.add(MaxPooling2D((2, 2)))
   model.add(Dropout(0.2))
   model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Dropout(0.2))
   model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
   model.add(MaxPooling2D((2, 2)))
   model.add(Dropout(0.2))
   model.add(Flatten())
   #connected layers
   model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
   model.add(Dropout(0.5))
   model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
   model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
   model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
   model.add(Dense(1, activation='sigmoid'))
   # Model summary
   print(model.summary())
   plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
   # compile model
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   return model

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
   pyplot.savefig('model_loss_plot.png')
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
   pyplot.savefig('model_accuracy_plot.png')
   pyplot.close()
   loss_curve(history)

# run the data
def run_test():
   # define model
   model = define_model()

   # create data generators
   train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)
   test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

   # prepare iterators
   train_it = train_datagen.flow_from_directory('/Users/blaiseswartwood/Downloads/Dataprep2/Spectrogram/train',
                                                class_mode='binary', batch_size=64, target_size=(200, 200))
   test_it = test_datagen.flow_from_directory('/Users/blaiseswartwood/Downloads/Dataprep2/Spectrogram/test',
                                              class_mode='binary', batch_size=64, target_size=(200, 200))
   # fit model
   history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                 validation_data=test_it, validation_steps=len(test_it), epochs=25, verbose=1)
   # evaluate model
   _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
   print('> %.3f' % (acc * 100.0))

   # learning curves
   accuracy_curve(history)
   # save model
   model.save('final_model3.h5')


# entry point, run the test
run_test()

