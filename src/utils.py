from enum import Enum
import numpy as np
from PIL import Image


def read_img(path):
    img = Image.open(path)
    img_array = np.array(img)

    return img_array

class OutputPath:
    HE = "images/he.png"
    AHE = "images/ahe.png"
    WHE = "images/whe.png"
    EIHE = "images/eihe.png"
    MMSICHE = "images/mmsiche.png"
    CLAHE = "images/clahe.png"
    ACLAHE = "images/aclahe.png"
    
class Algorithm(str, Enum):
	HE = "he"
	AHE = "ahe"
	WHE = "whe"
	EIHE = "eihe"
	MMSICHE = "mmsiche"
	CLAHE = "clahe"
	ACLAHE = "aclahe"