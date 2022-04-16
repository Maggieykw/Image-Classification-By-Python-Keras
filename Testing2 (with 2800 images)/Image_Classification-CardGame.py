
# coding: utf-8

# In[2]:


#Before training model, test it
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/cardgame/pic_001.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cardgame', save_format='jpeg'):
    i += 1
    if i > 20:
        break 


# In[2]:


'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


#optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              #optimizer=optimizer,
              #optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

H = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

modelName = "CardGame.model"
model.save(modelName)
model.save_weights('first_try.h5')

group1 = "Card Game"
group2 = "Not Card Game"

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on " + group1 + "/" + group2)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


# In[9]:


from keras.optimizers import RMSprop


# In[ ]:


#tutorial: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

#Code for training a model
#https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

#Code for fine-tuning a model (didn't use it)
#https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975


# In[3]:


loss = [0.7621,
0.6582,
0.6256,
0.6159,
0.5926,
0.5917,
0.5784,
0.5332,
0.5211,
0.5342,
0.5375,
0.5150,
0.5273,
0.5400,
0.5118,
0.5180,
0.5089,
0.5166,
0.4958,
0.4814]

val_loss = [0.7920,
0.6642,
0.6436,
0.6369,
0.6136,
0.6044,
0.5942,
0.5965,
0.5603,
0.5549,
0.5459,
0.5570,
0.5783,
0.5429,
0.5329,
0.5352,
0.5503,
0.5235,
0.5267,
0.4923]

acc = [0.5610,
0.6400,
0.6925,
0.7040,
0.7165,
0.7320,
0.7310,
0.7303,
0.7285,
0.7535,
0.7626,
0.7520,
0.7540,
0.7805,
0.7635,
0.7660,
0.7735,
0.7805,
0.7925,
0.8025]

val_acc = [0.5120,
0.5923,
0.6139,
0.6504,
0.6842,
0.7035,
0.7123,
0.7275,
0.7491,
0.7287,
0.7259,
0.7186,
0.7193,
0.7233,
0.7205,
0.7493,
0.7395,
0.7458,
0.7793,
0.7938]

group1 = "Board Game"
group2 = "Not Board Game"
plt.style.use("ggplot")
plt.figure()
N = 20
plt.plot(np.arange(0, N), loss, label="train_loss")
plt.plot(np.arange(0, N), val_loss, label="val_loss")
plt.plot(np.arange(0, N), acc, label="train_acc")
plt.plot(np.arange(0, N), val_acc, label="val_acc")
plt.title("Training Loss and Accuracy on " + group1 + "/" + group2)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_testing.png")


# In[27]:


loss = [0.753,
0.6527,
0.6325,
0.6159,
0.5868,
0.58,
0.5636,
0.5323,
0.535,
0.5234,
0.5469,
0.5496,
0.5244,
0.5196,
0.519,
0.516,
0.5155,
0.5131,
0.4705,
0.4686]

val_loss = [0.8013,
0.6783,
0.6760,
0.6468,
0.6414,
0.6029,
0.6135,
0.6010,
0.5930,
0.5915,
0.5826,
0.5732,
0.5931,
0.5429,
0.5329,
0.53,
0.531,
0.5245,
0.4931,
0.4823]

acc = [0.5338,
0.6446,
0.6678,
0.6981,
0.6839,
0.7258,
0.7228,
0.7303,
0.7384,
0.7369,
0.759,
0.7495,
0.7687,
0.7454,
0.754,
0.779,
0.791,
0.8254,
0.853,
0.8624]

val_acc = [0.512,
0.6103,
0.6062,
0.68,
0.705,
0.705,
0.7225,
0.7275,
0.7213,
0.7287,
0.7376,
0.7257,
0.7453,
0.7482,
0.7513,
0.7641,
0.7752,
0.7945,
0.8291,
0.8467]

group1 = "Card Game"
group2 = "Not Card Game"
plt.style.use("ggplot")
plt.figure()
N = 20
plt.plot(np.arange(0, N), loss, label="train_loss")
plt.plot(np.arange(0, N), val_loss, label="val_loss")
plt.plot(np.arange(0, N), acc, label="train_acc")
plt.plot(np.arange(0, N), val_acc, label="val_acc")
plt.title("Training Loss and Accuracy on " + group1 + "/" + group2)
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_testing.png")

