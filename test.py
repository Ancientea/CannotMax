import cv2
import numpy as np

def calculate_average_green(image_path, region):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None

    height, width, _ = image.shape
    x1, y1, x2, y2 = region
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

    region = image[y1:y2, x1:x2]
    green_channel = region[:, :, 1]
    average_green = np.mean(green_channel)
    return average_green

region = (0.3406, 0.5759, 0.4182, 0.6194)
image_paths = ["images/process/3.png", "images/process/4.png"]

for image_path in image_paths:
    avg_green = calculate_average_green(image_path, region)
    if avg_green is not None:
        print(f"Average green value in {image_path}: {avg_green}")