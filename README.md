# Helmet-Detection-Yolov10

## Description:
A project using YOLOv10 for helmet detection task on an input image, and a user-friendly interface with streamlit.
YOLOv10 (https://github.com/THU-MIG/yolov10).
Some information:
- Folder data store custom dataset (see more information inside it).
- Folder model store the best weight on the custom dataset, which trained by GPU-T4 on Colab.
## Demo:
<image src="./data/forReadMe.png">

## Installation:
NOTE: YOLOv10 can be set up from https://github.com/THU-MIG/yolov10 (should read this repo)

1. Clone this repo:
```bash
git clone https://github.com/LapTruongQuang/Helmet-Detection-Yolov10.git
```

2. Install the requirement.txt:
```bash
pip install -r requirement.txt
```

3. run (through streamlit):
```bash
streamlit run object_detection.py
```
