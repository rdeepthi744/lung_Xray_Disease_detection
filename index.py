import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import img_to_array
import pickle
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image
import sqlite3
import shutil
import telepot
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/developer.html')
def developer():
    return render_template('developer.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/accuracy_plot.png',
             'http://127.0.0.1:5000/static/loss_plot.png',
              'http://127.0.0.1:5000/static/confusion_matrix.png']
    content=['Accuracy Graph',
             "Loss Graph"
             'Confusion Matrix']

            
    
        
    return render_template('graph.html',images=images,content=content)
    


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                     [-1, 9,-1],
                                    [-1,-1,-1]])

        # # apply the sharpening kernel to the image
        sharpened =cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg',sharpened)

       
        
        
        
        model=load_model('Lungdisease_classifier.h5')
        path='static/images/'+fileName


        # Load the class names
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        Tre=""
        Tre1=""
        # Function to preprocess the input image
        def preprocess_input_image(path):
            img = load_img(path, target_size=(150,150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            return img_array

        # Function to make predictions on a single image
        def predict_single_image(path):
            input_image = preprocess_input_image(path)
            prediction = model.predict(input_image)
            print(prediction)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction[0][predicted_class_index]

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
                
            return predicted_class, confidence 

        predicted_class, confidence = predict_single_image(path)
        #predicted_class, confidence = predict_single_image(path, model, class_names)
        print(predicted_class, confidence)
        if predicted_class =="Cancer":
            str_label = "Cancer"
            Tre="Medical Treatmernt"
            Tre1=["chemotherapy, or radiation, depending on the cancer type.",
            "Healthy Lifestyle: Maintain proper nutrition, rest, and emotional support to aid recovery.",
            "Regular Monitoring: Attend follow-ups to track progress and adjust treatment as needed."]
                            
           
        elif predicted_class == 'Covid19':
            str_label = "Covid19"
            Tre="Medical Treatmernt"
            Tre1=["Rest and Hydration: Ensure plenty of rest and drink fluids to stay hydrated.",
            "Symptom Management: Use paracetamol for fever and body aches; lozenges or cough syrup for sore throat and cough.",
            "Isolation: Stay isolated to prevent spreading the infection to others."]


        elif predicted_class == 'Normal':
            str_label = "Normal"

            

        elif predicted_class == 'Pneumonia':
            str_label = "Pneumonia"
            Tre="Medical Treatmernt"
            Tre1=["Rest and Hydration: Get plenty of rest and drink fluids to help loosen mucus in the lungs.",
            "Medication: Take prescribed antibiotics if bacterial pneumonia is suspected, along with fever reducers like paracetamol.",
            "Symptom Monitoring: Watch for worsening symptoms, such as difficulty breathing, and consult a doctor if needed."]
                        

        elif predicted_class == 'Tuberculosis':
            str_label = "Tuberculosis"
            Tre="Medical Treatmernt"
            Tre1=["Take prescribed antibiotics (e.g., isoniazid, rifampin)",
            "Rest and Nutrition: Ensure adequate rest and a nutritious diet to support recovery.",
            "Regular Monitoring: Attend follow-up appointments to track progress and adjust treatment if needed."]
                        

       
            
        accuracy = f"The predicted image is {str_label} with a confidence of {confidence:.2%}"
        bot = telepot.Bot("8021608854:AAEpVpS079jGoT8oWchLi84pnaelIagLT_U")
        bot.sendMessage("1265586110",str_label + "detected" )
       
        bot.sendMessage("1265586110",Tre1)
         
            

       


        return render_template('results.html', status=str_label,accuracy=accuracy,Treatment=Tre,Treatment1=Tre1, ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg")
        
    return render_template('userlog.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
