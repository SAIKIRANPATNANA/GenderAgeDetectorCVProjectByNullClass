import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np 
import tensorflow as tf 

Model = tf.keras.models.load_model('/home/user/Documents/ML_DL_PROJECTS/GenderAgeDetectorCVProjectByNullClass/Age_N_Gender_Detector.h5')
top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detection')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
label2 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect(file_path):
    global label1, label2
    image = Image.open(file_path)
    image = image.resize((48, 48))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = np.delete(image, 0, 1)
    image = np.resize(image, (48, 48, 3))
    print(image.shape)
    genders = ['Male', 'Female']
    image = np.array([image]) / 255 
    predicted = Model.predict(image)
    age = int(np.round(predicted[1][0]))
    gender = int(np.round(predicted[0][0]))
    print('Predicted Age: ', str(age))
    print('Predicted Gender: ', str(gender))
    label1.configure(foreground='#011638', text=age)
    label2.configure(foreground='#011638', text=genders[gender])

def show_detect_button(file_path):
    detect_button = Button(top, text='Detect Image', command=lambda: detect(file_path), padx=10, pady=5)
    detect_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try: 
        file_path = filedialog.askopenfilename()
        uploaded_image = Image.open(file_path)
        uploaded_image.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        image = ImageTk.PhotoImage(uploaded_image)
        sign_image.configure(image=image)
        sign_image.image = image 
        label1.configure(text='')
        label2.configure(text='')
        show_detect_button(file_path)
    except:
        pass 

upload = Button(top, text='Upload an Image', command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
label2.pack(side='bottom', expand=True)
heading = Label(top, text='Age & Gender Detection', pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
