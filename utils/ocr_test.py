import cv2
import pytesseract
import numpy as np
from paddleocr import PaddleOCR
# model = YOLO("/app/runs/detect/train12/weights/best.pt")

# image_path = "/app/datas/test_imgs/113學測-1.png"

# image = cv2.imread(image_path)

# results = model(image)

# for result in results:
#     for box in result.boxes.xyxy:
#         x1, y1, x2, y2 = map(int, box)
#         cropped = image[y1:y2, x1:x2]

#         if (image.shape) == 3 and image.shape[2]==3:
#             gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#         else:
#             gray=cropped
#         processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)

#         processed = cv2.fastNlMeansDenoising(processed, h=30)

#         text = pytesseract.image_to_string(processed, lang="chi_tra")
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, text.strip(), (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         print(f"Detected Text: {text.strip()}")
        
# cv2.imshow("YOLO + OCR", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def ocr_tool(image:np.ndarray):
    # pytesseract need to 
    # apt install -y tesseract-ocr tesseract-ocr-chi-tra
    print(f"image.shape:{image.shape}")
    def ensure_grayscale(image):    
        if len(image.shape) == 3:
            return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    
    gray = ensure_grayscale(image)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    processed = cv2.fastNlMeansDenoising(processed, h=30)
    text = pytesseract.image_to_string(processed, lang="chi_tra",config="--psm 6")
    return text

def ocr_tool2(image:np.ndarray):
    # paddleocr need install
    # pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    print(f"image.shape:{image.shape}")
    def ensure_grayscale(image):    
        if len(image.shape) == 3:
            return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    
    gray = ensure_grayscale(image)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    processed = cv2.fastNlMeansDenoising(processed, h=30)
    ocr=PaddleOCR(use_angle_cls=True,lang="ch")
    text=ocr.ocr(image,cls=True)
    return text


if __name__=="__main__":
    image_path = "/app/datas/test_imgs/113學測-1.png"

    image = cv2.imread(image_path)

    print(ocr_tool2(image))
