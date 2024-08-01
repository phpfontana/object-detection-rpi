import numpy as np
import cv2
from PIL import Image
from models.yolo.model import YOLO
from models.yolo.utils import draw_boxes

def main():
    WEIGHTS = "src/models/yolo/weights/yolov9-t-converted-fp32.onnx"
    DEVICE = "cpu"
    IMG_PATH = "src/assets/test.png"

    img = Image.open(IMG_PATH)
    img = np.array(img)

    model = YOLO(WEIGHTS, DEVICE)

    detections = model.predict(img)

    img_with_boxes = draw_boxes(img, detections)

    # Display the image with bounding boxes
    cv2.imshow("Detections", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image with bounding boxes
    output_path = "src/outputs/output.png"
    cv2.imwrite(output_path, img_with_boxes)

    print(detections)


if __name__ == "__main__":
    main()
