import tensorflow as tf
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np

# EfficientDet modelini TensorFlow Model Zoo'dan yükle
detect_fn = tf.saved_model.load("efficientdet_d0_coco17_tpu-32/saved_model")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'][0].numpy(),
        detections['detection_classes'][0].numpy().astype(int),
        detections['detection_scores'][0].numpy(),
        category_index=None,
        use_normalized_coordinates=True,
        line_thickness=2,
    )

    cv2.imshow("EfficientDet", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
