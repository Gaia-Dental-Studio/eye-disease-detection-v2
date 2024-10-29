import tensorflow as tf
import cv2

# define croping function with tensorflow resize
def crop_image(image_path):
    image_data = tf.keras.preprocessing.image.load_img(image_path)

    array = tf.keras.preprocessing.image.img_to_array(image_data)
    image = tf.image.resize(
                            array, [200,200],
                            method='bilinear',
                            preserve_aspect_ratio=True,
                            antialias=False,
                            )
    # image = image / 255.0
    return image

#define method for image resize, croping and image Contrast Limited Adaptive Histogram Equalization (CLAHE)
#using Opencv 4

def image_resize(image_path, dim):
    img = cv2.imread(image_path)
    if img.shape[1] != img.shape[0]:
        x = img.shape[1]//2
        y = img.shape[0]//2
        x = x-y
        img = img[0:0+img.shape[0], x:x+img.shape[0]]
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def CLAHE(image_path, dim, clipLimit, tileGridSize):
    img = image_resize(image_path, dim)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2,a,b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def has_disease(text, disease_name):
    if disease_name in text:
        return 1
    else:
        return 0
    
def get_spesific_class(dataset, disease_name, disease_code):
    df = dataset.copy()
    
    w_disease = df[]

    return w_disease
