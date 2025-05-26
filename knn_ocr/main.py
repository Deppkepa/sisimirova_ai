import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

train_dir = "task/train/"
test_images = "task/"
min_region_size = 250
knn_neighbors = 2
space_threshold = 30


def formation_features(region):
    image = region.image
    height, width = image.shape  # строки, столбцы

    image_size = height * width  # всего пикселей
    area = image.sum() / image_size  # площадь фигуры
    perimeter = region.perimeter / image_size  # периметр фигуры
    wh_ratio = height / width if width > 0 else 0

    euler = region.euler_number  # дырки
    eccentricity = region.eccentricity * 8 if hasattr(region,
                                                      'eccentricity') else 0  # насколько фигура вытянутая растянутая
    wide_rows = (np.mean(image, axis=1) > 0.85).sum() > 2  # Показывает заполненость 1 по строкам
    hole_size = image.sum() / region.filled_area if region.filled_area > 0 else 0  # помогает определять дырки а именно плотность 0 пикселей относительно заполненой площади области
    solidity = region.solidity * 2 if hasattr(region, 'solidity') else 0  # выпуклость букв

    return np.array([
        area, perimeter, euler, eccentricity, hole_size, wh_ratio, solidity, wide_rows * 4

    ])


def get_area(region):
    return region.area


def preparation_training_data(training_dir):
    features, labels_list = [], []

    for label_idx, label_name in enumerate(os.listdir(training_dir)):
        label_path = os.path.join(training_dir, label_name)
        if not os.path.isdir(label_path):
            continue

        for image_path in glob.glob(os.path.join(label_path, "*.png")):

            try:
                image = plt.imread(image_path)
                if image.ndim == 3:
                    image = image.mean(axis=2)  # Перевести в серое
                if image is None:
                    print(f"Warning: Could not load image {image_path}")
                    continue

                binary_image = (image > 0).astype(int)

                labeled_image = label(binary_image)
                regions = regionprops(labeled_image)

                if not regions:
                    print(f"Warning: No regions found in image {image_path}")
                    continue

                feature_vector = formation_features(max(regions, key=get_area))
                features.append(feature_vector)
                labels_list.append(label_idx)

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue

    return np.array(features), np.array(labels_list)


def get_centroid_x(region):
    return region.centroid[1]


def recognize_text(image_path, knn, labels):
    if not os.path.isfile(image_path):
        print(f"File {image_path} does not exist.")
        return ""

    image = plt.imread(image_path)
    if image.ndim == 3:
        image = image.mean(axis=2)  # Перевести в серое

    if image is None:
        print(f"Failed to upload image along the way: {image_path}")
        return ""
    binary_image = (image > 0).astype(int)

    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    regions = sorted(regions, key=get_centroid_x)  # сортируем по x чтобы правильно формировать строку

    result_text = ""
    last_char_right = 0
    for i, region in enumerate(regions):
        if np.sum(region.image) > min_region_size:
            feature_vector = formation_features(region)
            feature_vector = feature_vector.reshape(1, -1).astype(np.float32)
            ret, results, neighbors, dist = knn.findNearest(feature_vector, knn_neighbors)
            current_char_left = region.bbox[1]
            if i > 0 and (current_char_left - last_char_right) > space_threshold:
                result_text += " "
            result_text += labels[int(ret)][-1]  # формуруем строку из предсказанных символов
            last_char_right = region.bbox[3]
    return result_text


train_features, train_targets = preparation_training_data(train_dir)

if train_features.size > 0 and train_targets.size > 0:
    knn = cv2.ml.KNearest_create()
    train_targets = train_targets.astype(np.float32).reshape(-1, 1)
    knn.train(train_features.astype(np.float32), cv2.ml.ROW_SAMPLE, train_targets)
else:
    print("No training data loaded. Cannot train KNN.")
    exit()

labels = os.listdir(train_dir)

image_files = [f for f in os.listdir(test_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
for i, image_file in enumerate(image_files):
    image_path = os.path.join(test_images, image_file)
    recognized_text = recognize_text(image_path, knn, labels)
    print(f"{i}.png: {recognized_text}")
