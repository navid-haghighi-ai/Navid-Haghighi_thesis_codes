import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

#Definition of function for image prediction
def predict_image():
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    image = image.resize((130, 130))  #Change the size of the input image of the model
    image = np.array(image) / 255.0  #Image scaling
    image = image.reshape((1, 130, 130, 3))  #Transform the image to feed the model

    #Prediction using the model.
    prediction = model.predict(image)
    result_label.config(text=f"Result: {'No malaria parasites were found.' if prediction[0][0] > 0.5 else 'The sample has malaria parasite.'}")

    #The main window to display the result.
    image = Image.open(file_path)
    image = image.resize((130, 130))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

#Setting of the graphical interface.
root = tk.Tk()
root.title("Malaria diagnosis-By Navid Haghighi")
root.geometry("400x400")

#Background image
background_image = Image.open("/Software-Background.jpg")
background_image = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)

open_button = tk.Button(root, text="Choose File", command=predict_image, relief="raised", bd=3)
open_button.pack(pady=20)

result_label = tk.Label(root, text="Please enter a blood sample.", font=("Helvetica", 14), bg="#ffffff", fg="#363535")
result_label.pack()

#Selected image
image_label = tk.Label(root)
image_label.pack()

#Software Icon
icon = Image.open("/Malaria-Icon.png")
icon = ImageTk.PhotoImage(icon)
root.iconphoto(False, icon)

#Loading the model (trained weight of the main model)
model = tf.keras.models.load_model('/MobileNetV2_new_malaria.h5')

root.mainloop()
