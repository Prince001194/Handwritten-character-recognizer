import cv2
import numpy as np
import os
from keras.models import load_model

model = load_model('models/handwritten_character_recog_model.h5')

words = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N', 24:'O',25:'P'
	 ,26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}


def resize_image(image,height=None,width=None):

    if height is None:
        ratio = float(width/image.shape[1])
        image = cv2.resize(image,(width,int(image.shape[0]*ratio)))
    else:
        ratio = float(height/image.shape[0])
        image = cv2.resize(image,(int(image.shape[1]*ratio),height))
    
    return image


def predict_char(image):

    image = resize_image(image,width=700)
    if(image.shape[0] > 600):
        image = resize_image(image,height=600)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts,_ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 10) and (h >= 15) and (min(h,w)/max(h,w) > 0.2):
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            
            (tH, tW) = thresh.shape
            if tW > tH:
                thresh = resize_image(thresh,width=28)
            else:
                thresh = resize_image(thresh,height=28)

            (tH, tW) = thresh.shape
            dX = int(max(0, 40 - tW) / 2.0)
            dY = int(max(0, 40 - tH) / 2.0)
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))

            chars.append((padded, (x, y, w, h)))

    padding = (int(image.shape[0]*0.2),int(image.shape[1]*0.2))
    image = cv2.copyMakeBorder(image, top=padding[0], bottom=padding[0],
                left=padding[1], right=padding[1], borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    if(chars.size == 0):
        return image
    
    preds = model.predict(chars)

    
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        x = x+padding[1]
        y = y+padding[0]
        i = np.argmax(pred)
        prob = pred[i]
        label = words[i]
        print("[INFO] {} - {:.2f}%".format(label, prob * 100))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    
    return image


if __name__ == "__main__":

    for filename in os.listdir(os.path.join(os.getcwd(),"test_images")):
        image = cv2.imread(os.path.join(os.getcwd(),"test_images",filename))
        image = predict_char(image)
        cv2.imshow(filename,image)
        cv2.waitKey()
        cv2.destroyAllWindows()