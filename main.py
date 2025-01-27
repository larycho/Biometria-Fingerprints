import cv2
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Otwieranie obrazu
path = "Fingerprints/1.bmp"

file_extension = pathlib.Path(path).suffix

if (file_extension in ['.jpg', '.tif', '.bmp','.png']):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Binaryzacja Otsu
threshold, image_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
# Filtr medianowy
median = cv2.medianBlur(image_otsu, 5) 

# Otwiera okno w przeglądarce z możliwością oglądania, zoomowania zdjęcia, sprawdzania wartości pikseli itp.
fig = px.imshow(cv2.cvtColor(median, cv2.IMREAD_GRAYSCALE))
fig.show()