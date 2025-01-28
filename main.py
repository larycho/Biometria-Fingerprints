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
                                                        "*.png")))
    
    # Otwieranie obrazu
#    path = "thin2.png"
    file_extension = pathlib.Path(path).suffix
    image = None

    if file_extension in ['.jpg', '.tif', '.bmp', '.png']:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return

    image_preprocessed = preprocess(image)
    image_thinned = k3m(image_preprocessed)
    image_cn = crossing_number(image_thinned)

    minutiae_image = mark_minutiae(image_thinned, image_cn)
    browser_display(minutiae_image)


def preprocess(image):
    threshold, image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # Binaryzacja Otsu
    median = cv2.medianBlur(image_otsu, 5) # Filtr medianowy
    return median


def browser_display(image):  # Otwiera okno w przeglądarce z możliwością oglądania, zoomowania zdjęcia, sprawdzania wartości pikseli itp.
    fig = px.imshow(cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE))
    fig.show()


def k3m(image):
    def count_neighbors(neighbourhood):
        return np.sum(neighbourhood) - neighbourhood[1, 1]

    def transitions(neighbourhood):
        neighbours = neighbourhood.flatten()[np.array([1, 2, 5, 8, 7, 6, 3, 0])]
        return np.sum((neighbours[:-1] == 0) & (neighbours[1:] == 1))

    removal_table = [
        {3}, {3, 4}, {3, 4, 5}, {3, 4, 5, 6}, {3, 4, 5, 6, 7}
    ]

    rows, cols = image.shape
    inverted_binary_image = (image == 0).astype(np.uint8)
    anything_removed = True
    while anything_removed:
        anything_removed = False
        for phase, removal_number_set in enumerate(removal_table):
            to_be_removed = []

            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    if inverted_binary_image[row, col] == 0:
                        continue

                    neighborhood = inverted_binary_image[row - 1:row + 2, col - 1:col + 2]
                    neighbors = count_neighbors(neighborhood)

                    # Warunki usunięcia: aktywni sąsiedzi, liczba przejść i tabele dla fazy
                    if (neighbors in removal_number_set
                            and 2 <= neighbors <= 6
                            and transitions(neighborhood) == 1):
                        to_be_removed.append((row, col))

            # Usuwanie pikseli wskazanych w bieżącej fazie
            for row, col in to_be_removed:
                inverted_binary_image[row, col] = 0

            if to_be_removed:
                anything_removed = True

    # Konwersja do białego tła (255) i czarnych linii
    thinned_image = (1 - inverted_binary_image) * 255

    return thinned_image


def crossing_number(image):
    y = image.shape[0]
    x = image.shape[1]
    new_image = np.zeros((y,x), dtype = 'uint8')
    image = image / 255

    for i in range(1, y - 1): # Przejscie po obrazie
        for j in range(1, x - 1):
            # Tablica z wartościami p1 do p9
            p = [ image[i + 1][j], image[i + 1][j + 1], image[i][j + 1], image[i - 1][j + 1], 
                 image[i - 1][j], image[i - 1][j - 1], image[i][j - 1], image[i + 1][j-1], image[i + 1][j] ]

            sum = 0 
            
            # Obliczanie sumy różnic |pi - pi+1|
            for k in range(8):
                sum += abs(p[k] - p[k + 1])
            sum = sum / 2
            # Nadanie pikselowi wartości równej wynikowi algorytmu dla niego
            new_image[i][j] = int(sum)
            
    return new_image


def mark_minutiae(image_og, image_cn):
    y = image_cn.shape[0]
    x = image_cn.shape[1]
    image = cv2.cvtColor(image_og,cv2.COLOR_GRAY2RGB)

    for i in range(1, y - 1):  # Przejscie po obrazie
        for j in range(1, x - 1):

            if image_og[i][j] == 0:
                match image_cn[i][j]:
                    case 3: # Wykryto rozwidlenie - rysuje kwadracik 3 x 3 o kolorze zielonym
                        # for k in (0,1):
                        #     for l in (0,1):
                        #         image[i + k][j + l] = [0,255,0]
                        image[i][j] = [0,255,0]
                    case 1: # Wykryto zakonczenie krawedzi - Rysuje kwadracik 3 x 3 o kolorze czerwonym
                        # for k in (0,1):
                        #     for l in (0,1):
                        #         image[i + k][j + l] = [255,0,0]
                        image[i][j] = [255, 0, 0]
            
    return image


app = customtkinter.CTk()
app.geometry("200x200")
app.title("Odciski palców")

Button(app,
       text="Browse Files",
       command=browse_files).place(relx=0.5, rely=0.5, anchor='center')

app.mainloop()
