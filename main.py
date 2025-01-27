import cv2
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

import tkinter
import customtkinter
from tkinter import filedialog
from tkinter import Button, Label

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

# Function for opening the file explorer window
def browse_files():
    path = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("JPEG files",
                                                        "*.jpg"),
                                                        ("BMP files",
                                                        "*.bmp"),
                                                        ("TIFF files",
                                                        "*.tif"),
                                                        ("PNG files",
                                                        "*.png"),
                                                       ("all files",
                                                        "*.*")))  
    
    # Otwieranie obrazu
    file_extension = pathlib.Path(path).suffix

    if (file_extension in ['.jpg', '.tif', '.bmp','.png']):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    image_preprocessed = preprocess(image)
    browser_display(image_preprocessed)


def preprocess(image):
    threshold, image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # Binaryzacja Otsu
    median = cv2.medianBlur(image_otsu, 5) # Filtr medianowy
    return median

def browser_display(image):     # Otwiera okno w przeglądarce z możliwością oglądania, zoomowania zdjęcia, sprawdzania wartości pikseli itp.
    fig = px.imshow(cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE))
    fig.show()

app = customtkinter.CTk()
app.geometry("200x200")
app.title("Odciski palców")

button_explore = Button(app, 
                        text = "Browse Files",
                        command = browse_files).place(relx=0.5, rely=0.5, anchor='center')

app.mainloop()
