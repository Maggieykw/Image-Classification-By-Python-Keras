
# coding: utf-8

# In[ ]:


###Overview:
#[1] using anaconda (which installed TensorFlow)
#[2] train a model for image classification
#[3] process image classification of selected product (from SQL query result)
#[4] store result into a csv file



###For training model
#Before running code:
#[1] download train_network.py if you want to run function of "train_network_sample" (for running sample from online: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/)
#[2] open a file called "images" (inside working directory) containing two subfiles (rename them as two groups for classification; "boardgame" & "not_boardgame" in this case) 
#[3] collect large quantity of pictures (two groups for classification) by Fatkun (https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf)
#[4] put the pictures into corresponding subfiles (500 for each subfiles in this case)

#Required file: 
#[1]lenet.py
#[2]train_network.py; 
#[3]file called "images" having two subfiles (one for excluded group; one for not excluded group; containing certain number of picture inside each subfile)

#Running code: 
#[1] import required module (making sure it is installed)
#[2] set directory to the file storing related files
#[3] run lenet.py
#[4] run function of "train_network_sample" OR function of "train_network(dataset,ExcludedGroup)"

#Output (inside working directory): 
#[1] a graph of training loss and accuracy on two groups; 
#[2] BoardGame_Not_BoardGame.model (in this case; will be used for testing)

#Remark:
#I used function of "train_network_sample" to test if the online code works
#hence, it is better to use function of "train_network(dataset,ExcludedGroup)" for our image classification




###For Image Classification (=testing model)
#Before running code:
#[1] run the above code for training model to create a model
#[2] select field "name" & "imgs" from top100 table in SQL (see attached code 1)
#[3] export the query result into "query_result.csv"
#[4] inside the csv file, expand column of "imgs" to columns(-->data-->text to columns) 
#[5] run VBA for "query_result.csv" (see attached code 2) as we want to one column of name and one column of url instead of multiply columns of url each row of name

#Required file: 
#[1] BoardGame_Not_BoardGame.model (in this case; will be used for testing)
#[2] query_result.csv

#Running code: 
#[1] import required module (making sure it is installed)
#[2] run all the function mentioned below (run main function at last)

#Output (inside working directory): 
#[1] img01.jpg (for working, will be deleted at the end)
#[2] ImageClassiferResult.csv (containing product name, img url, classified group & similarity)

#Remark:
#None


# In[6]:


#Upgrade pip
get_ipython().system('pip install --upgrade pip')


# In[ ]:


#Install required module to run function called train_network(OR train_network.py)
import sys

get_ipython().system('{sys.executable} -m pip install h5py')
get_ipython().system('{sys.executable} -m pip install keras')
get_ipython().system('{sys.executable} -m pip install matplotlib')
get_ipython().system('{sys.executable} -m pip install -U scikit-learn')
get_ipython().system('{sys.executable} -m pip install imutils')
get_ipython().system('{sys.executable} -m pip install opencv-python')


# In[14]:


#Install extra module to run function called test_network (OR test_network.py)
import sys

get_ipython().system('{sys.executable} -m pip install image ')


# In[19]:


#Install extra module to run function called rawdata
import sys

get_ipython().system('{sys.executable} -m pip install pandas')


# In[ ]:


import sys
get_ipython().system('reload(sys)')
get_ipython().system("sys.setdefaultencoding('utf8')")

#Used to solve error "Matplotlib is building the font cache using fc-list mac python" when running train_network.py
#https://askubuntu.com/questions/788487/when-trying-to-import-matplotlib-pyplot-get-unicodedecodeerror


# In[1]:


# import the necessary packages (rawdata)
import pandas as pd
import numpy as np

# import the necessary packages (train_network)
import matplotlib
matplotlib.use('Agg')
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from pyimagesearch.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
#import numpy as np #mentioned above
import argparse
import random
import cv2
import os #also for remove img

#import the necessary packages (convertGIFtoJPG)
from PIL import Image
import requests

# import the necessary packages (test_network)
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#import numpy as np #mentioned above
#import argparse #mentioned above
import imutils
#import cv2 #mentioned above
import urllib.request, io

#import the necessary packages (downloadIMGfromURL)
#import urllib.request #mentioned above

#import the necessary packages (csv writing and appending)
import csv
from collections import defaultdict


# In[67]:


#Set directory to the file storing train_network.py
get_ipython().run_line_magic('cd', '/Users/kawaiyim/FYP/Image classification:recognition/3rd Method(Keras)/image-classification-keras_FYP')


# In[2]:


get_ipython().run_line_magic('run', 'lenet.py #For training NN')


# In[9]:


def rawdata():
    df = pd.read_csv('query_result.csv')
    ListOfProductName = df['name'].tolist()
    ListOfURL = df['imgs'].tolist()
    
    return(ListOfProductName,ListOfURL)


# In[10]:


def Result_write_csv():
    with open("ImageClassiferResult.csv","w") as csv_file:
        writer = csv.writer(csv_file,delimiter=',')
        writer.writerow(["Product Name","img","Assigned Group","Similarity(%)"])


# In[52]:


#Dataset are derived from a file called "images" including two files storing images of each group respectively
def train_network_sample():
    get_ipython().run_line_magic('run', 'train_network.py --dataset images --model CardGame_Not_CardGame.model')

#train_network_sample()


# In[6]:


#Dataset are derived from a file called "images" including two files storing images of each group respectively
def train_network(group1,group2,dataset,ExcludedGroup):
    
    Excluded = ExcludedGroup
    NotExcluded = "Not_" + ExcludedGroup
    
    modelName = Excluded+"_"+NotExcluded+".model"

    #group1 = "board game"
    #group2 = "not board game"
    # set the matplotlib backend so figures can be saved in the background
    #matplotlib.use("Agg") #typed in the above code already
    
    # initialize the number of epochs to train for, initia learning rate,
    # and batch size
    EPOCHS = 10
    INIT_LR = 1e-3
    BS = 32

    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (28, 28))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == group1 else 0
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,
        labels, test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=28, height=28, depth=3, classes=2)
    #opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) #from original code but it is not suitable
    opt = SGD(lr=0.00001, momentum=0.4)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(modelName)

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on " + group1 + "/" + group2)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")
    
    return modelName

train_network("card game","not card game","images","CardGame")


# In[11]:


def convertGIFtoJPG(url,imgName):
    #convert url to gif
    uri = url
    with open('img01.gif', 'wb') as f:
        f.write(requests.get(url).content)

    #convert gif to jpg
    try:
        infile = 'img01.gif'
        outfile = imgName #'img01.jpg'

        Image.open(infile).convert('RGB').save(outfile)
        
        remove_img(infile)

    except IOError:
        print ("Cannot convert" ,infile)

#testing:
#url = 'https://ksr-ugc.imgix.net/assets/014/389/156/e0c5c001a0812a5cf82a56b9f2f80f2f_original.gif?w=680&fit=max&v=1478266603&auto=format&frame=1&q=92&s=f1bfa0c754658e00c4e4533edb56b027'


# In[12]:


def downloadIMGfromURL(url,imgName):
    try:
        #if the image is jpg format
        if "jpg" in url:
            urllib.request.urlretrieve(url,imgName) 
        else:
        #if the image is gif format --> additional work: convert it to jpg before image classification
            convertGIFtoJPG(url,imgName)
    except:
        print('Web site does not exist') 

#downloadIMGfromURL("https://ksr-ugc.imgix.net/assets/005/758/590/b5a0093a9ac8ff1f3ac6062809528b26_original.jpg?w=680&fit=max&v=1461065941&auto=format&q=92&s=973542e86a2f4cef497f2a728e522af5","img01.jpg") #for testing


# In[2]:


def test_network(group1,group2,modelName,imgName):
    
    #group1 = "board game"
    #group2 = "not board game"

    # load the image
    image = cv2.imread(imgName)
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(modelName)

    # classify the input image
    (notExcluded, Excluded) = model.predict(image)[0]

    # build the label
    result = []
    label = group1 if Excluded > notExcluded else group2
    proba = Excluded if Excluded > notExcluded else notExcluded

    prob=float("{0:.2f}".format(proba * 100))

    result.append(label)
    result.append(prob)
    
    label = "{}: {:.2f}%".format(label, proba * 100)

    # draw the label on the image
    #output = imutils.resize(orig, width=400)
    #cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)

    # show the output image
    #cv2.imshow("Output", output)
    #cv2.waitKey(0)
    
    return result

#test_network("BoardGame_Not_BoardGame.model","img01.jpg") #testing


# In[14]:


def Result_append_csv(data,ProductName,imgURL):
    with open("ImageClassiferResult.csv","a") as csv_file:
        result=[]
        if data:
                result.append(ProductName)
                result.append(imgURL)
                result = result + data
        else:
                result.append(ProductName)
                
        writer = csv.writer(csv_file)
        writer.writerow(result)   


# In[15]:


def remove_img(imgName):
# check if file exists or not
    if os.path.exists(imgName) is True:
        os.remove(imgName)
    else:
        # file did not exists
        pass

#remove_img("img01.jpg") #for testing


# In[4]:


def main():
    ListOfProductName = rawdata()[0]
    ListOfIMGS = rawdata()[1]
    
    Result_write_csv()
    
    #Used for function of train_network & test_network, change the name if necessary
    group1 = "card game"
    group2 = "not card game"
    
    #Train a model (did before runing code)
    #datasetFile = "images"
    #ExcludedGroup = "BoardGame"
    #modelName = train_network(group1,group2,datasetFile,ExcludedGroup)
    modelName = "CardGame_Not_CardGame.model" #Inside working directory
    
    x=0
    
    for url in ListOfIMGS:
        
        #Store the corresponding product name
        ProductName = ListOfProductName[x]

        #Store it as img01.jpg inside working directory after downloading picture from url
        imgName = "img01.jpg" 
        
        #Download the picture from url
        downloadIMGfromURL(url,imgName)
        
        #Image classification and write the result into csv
        Result_append_csv(test_network(group1,group2,modelName,imgName),ProductName,url)
        
        #Remove the picture
        remove_img(imgName)
        
        x+=1

main()


# In[ ]:


#tutorial: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/


# In[ ]:


test_network("card game","not card game","CardGame_Not_CardGame.model","testing1.jpg")

