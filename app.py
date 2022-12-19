# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:50:15 2022

@author: user
"""

from tensorflow.keras.models import load_model

import os
import cv2
import glob
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix ,classification_report

from flask import Flask, render_template, request

app = Flask(__name__)
model = load_model('trained_model/InceptionV3.h5')
target_img = os.path.join(os.getcwd() , 'static/videos')

@app.route('/')
def index_view():
    return render_template('index.html')

#Allow files with extension mp4, avi and 
ALLOWED_EXT = set(['mp4' , 'avi' , 'mkv'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
        
        
def video_to_Frames(input_video):
    test = []
    # frame
    currentframe = 0
    # Read the video from specified path
    cam = cv2.VideoCapture(input_video)
    try:
    
        # creating a folder named data
        if not os.path.exists('test'):
            os.makedirs('test')
    
        # if not created then raise error
    except OSError:
        print('Error: Creating directory of data')
    
    while(True):
    
        # reading from frame
        ret, frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            # save frame
            name = 'test/test' + str(currentframe) + '.jpg'
            print('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 30  # i.e. at 30 fps, this advances one second
            cam.set(1, currentframe)
            
            img = cv2.imread(name)
            # resize image by specifying custom width and height
            img = cv2.resize(img, (224, 224))
            test.append(img)
            
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return test


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/videos/', filename)
            file.save(file_path)
            data = video_to_Frames(file_path)
            
            data = np.array(data)
            predictions = np.argmax(model.predict(data), axis=1)
            
            action_per = (predictions[np.where(predictions == 0)].size)/len(predictions)
            horror_per = (predictions[np.where(predictions == 1)].size)/len(predictions)
            romantic_per = (predictions[np.where(predictions == 2)].size)/len(predictions)
            sci_fi_per = (predictions[np.where(predictions == 3)].size)/len(predictions)

            classes = ['Action', 'Horror', 'Romantic', 'Sci-Fi']
            plt.figure(1)
            plt.figure(figsize=(5, 5))
            plt.hist(predictions, edgecolor='black')
            plt.xlabel("Genre", fontsize=16)
            plt.ylabel("Frequency", fontsize=16)
            plt.title('Histogram of Movie Scenes', fontsize=20)
            plt.xticks([0, 1, 2, 3], classes, rotation=20)
            plt.savefig('static/Plots/bar_plot.png')


            y = np.array([action_per , horror_per, romantic_per, sci_fi_per])
           
            plt.figure(2)
            explode = (0, 0.1, 0, 0)
            plt.figure(figsize=(5, 5))
            plt.pie(y, labels = classes ,explode=explode, startangle=90, autopct='%1.1f%%', shadow=True)
            plt.legend(title = "Genres:")
            plt.savefig('static/Plots/pie_plot.png')
            
            
            def max_no_label(no):
                max_number = max(no)
                if max_number == action_per:
                    
                    genre = "Action"
                elif max_number == horror_per:
                    
                    genre = "Horror"
                elif max_number == romantic_per:
                    
                    genre = "Romantic"
                else:
                    
                    genre = "Sci-fi"
                
        
            max_no_label([action_per, horror_per , romantic_per , sci_fi_per])
    
    
            
            return render_template('predict.html', action_per = action_per, horror_per = horror_per, romantic_per = romantic_per , sci_fi_per = sci_fi_per , user_video = file_path , genre = genre)
        else:
            return "Unable to read the file. Please check file extension"

if __name__ == '__main__':
    app.run(debug=True)
