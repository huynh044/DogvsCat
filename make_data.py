import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Danh mục và đường dẫn dữ liệu
CATEGORIES = ["Cat", "Dog"]
DATA_PATH = "D:/All_learn_programs/Python/ML_and_DL/Test/DogvsCat/PetImages"
DATA_OUTPUT = "D:/All_learn_programs/Python/ML_and_DL/Test/DogvsCat/Data"
IMAGE_SIZE = 50
training_data = []

# Hàm tạo dữ liệu
def create_data():
    print("Start processing . . .")
    for category in CATEGORIES:
        path = os.path.join(DATA_PATH, category)
        label = CATEGORIES.index(category)
        print(f"Processing {category}'s dataset...")
        for image in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
                training_data.append([new_array, label])
            except Exception as e:
                pass
    print("Processing complete!")

# Gọi hàm tạo dữ liệu
create_data()

# Trộn ngẫu nhiên dữ liệu
import random
random.shuffle(training_data)

# Chuyển đổi dữ liệu thành X, Y
X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)


# Lưu dữ liệu vào file CSV
def save_to_csv(output_path):
    # Tạo DataFrame với pixel_data (dưới dạng chuỗi) và label
    data = {
        "pixel_data": [x.flatten().tolist() for x in X],  # Chuyển ảnh thành chuỗi danh sách
        "label": Y
    }
    df = pd.DataFrame(data)
    # Lưu file CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

# Đường dẫn file CSV đầu ra
csv_output_path = os.path.join(DATA_OUTPUT, "dog_vs_cat_data.csv")
save_to_csv(csv_output_path)
