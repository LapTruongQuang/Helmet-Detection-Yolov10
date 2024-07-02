import os
from ultralytics import YOLOv10
import cv2
import torch
import tqdm
from PIL import Image


def process_image(img):
    return cv2.cvtColor(img.plot(), cv2.COLOR_BGR2RGB)


class HelmetDetectionModel:
    def __init__(self, model_patch):
        self.model = YOLOv10(model_patch)

    def predict(self, image_patch):
        return self.model(source=image_patch)[0]


def load_model(model_patch):
    return HelmetDetectionModel(model_patch)


def run_model(image_patch, model):
    return model.predict(image_patch)


def run_dir(model, test_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_files = [f for f in os.listdir(
        test_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

    for img_file in tqdm(img_files,
                         desc='Processing images',
                         leave=True,
                         ncols=100):
        img_path = os.path.join(test_dir, img_file)
        img_out = model.predict(img_path)
        img_out.save(os.path.join(
            output_dir, f'{os.path.splitext(img_file)[0]}_pred.png'))

    print(f'Results are saved in {output_dir}.')


if __name__ == '__main__':
    MODEL_PATH = '../model/best.pt'
    TEST_DIR = '../data/test/images'
    OUTPUT_DIR = '../assets/output_images'
    model = load_model(MODEL_PATH)
    run_dir(model, TEST_DIR, OUTPUT_DIR)
