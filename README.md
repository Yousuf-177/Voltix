
---

# ⚡ Voltix — YOLO Multi-Class Object Detection (Hackathon Project)

This repository is developed by **Team Voltix** for the **Hack Of Thrones**.  
It implements a YOLO-based object detection model trained on the **Falcon Duality AI dataset** to identify seven safety-related objects.
 
The model is designed to detect **7 safety-related objects** from images and videos.

---

## 🧠 Dataset Overview

**Dataset Name:** Falcon Duality AI  
**Classes (`nc: 7`):**
```
['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
```

You can get the dataset for training as well as testing dataset on [**Falcon Duality AI**](https://falcon.duality.ai/secure/documentation/7-class-hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=hackofthrones)

---

## ⚙️ Environment Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Yousuf-177/Voltix.git
cd Voltix
```

### 2️⃣ Create a Virtual Environment (Recommended)
```bash
python -m venv yolovenv
source yolovenv/bin/activate      # On Linux/Mac
yolovenv\Scripts\activate         # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### Run Model on Custom Test Data
```bash
python yolo_detect.py --model yolov8s.pt --source usb0 --resolution 1280x720
```
Here are all the arguments for yolo_detect.py:

- `--model`: Path to a model file (e.g. `my_model.pt`). If the model isn't found, it will default to using `yolov8s.pt`.
- `--source`: Source to run inference on. The options are:
    - Image file (example: `test.jpg`)
    - Folder of images (example: `my_images/test`)
    - Video file (example: `testvid.mp4`)
    - Index of a connected USB camera (example: `usb0`)
    - Index of a connected Picamera module for Raspberry Pi (example: `picamera0`)
- `--thresh` (optional): Minimum confidence threshold for displaying detected objects. Default value is 0.5 (example: `0.4`)
- `--resolution` (optional): Resolution in WxH to display inference results at. If not specified, the program will match the source resolution. (example: `1280x720`)
- `--record` (optional): Record a video of the results and save it as `demo1.avi`. (If using this option, the `--resolution` argument must also be specified.)


---

## 🧩 Training the Model

To start model training:
```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt --epochs 100 --mosaic 0.50 --optimizer AdamW --momentum 0.9
```
You can modify `--epochs` `--mosaic` `--optimizer` `--momentum` as per your choices

This will automatically create the following directory:
```
runs/detect/train/
 ├── weights/
 │   ├── best.pt
 │   └── last.pt
 ├── results.png
 ├── confusion_matrix.png
 └── metrics.csv
```

---

## 🧪 Testing / Evaluating the Model

### Run Evaluation on Validation Set
```bash
yolo detect val model=runs/detect/train/weights/best.pt data=dataset/data.yaml
```

---

## 📊 Model Performance

| Metric | Value | Notes |
|:--------|:------:|:------|
| **mAP@0.5** |  | (Mean Average Precision at IoU threshold 0.5) |
| **Precision** |  | (Proportion of correct positive predictions) |
| **Recall** |  | (Proportion of actual positives correctly identified) |
| **Confusion Matrix** |  | (Matrix visualization of true vs predicted classes) |
| **Predictions (Sample)** |  | (Image samples with bounding boxes and labels) |



---

## 🔁 Reproducing Final Results

To reproduce the same model results:
1. Clone the repo and set up the environment as per [Environment Setup](#️-environment-setup).
2. Use the same dataset structure (`data.yaml`, train/val/test splits).
3. Train using the same configuration command.
4. Evaluate using the same validation command.
5. The results (metrics, weights, and logs) will be stored under `runs/detect/train/`.

---

## 🖼️ Expected Outputs

### During Training:
- `results.png` → shows training & validation loss, precision, recall, and mAP curves  
- `weights/best.pt` → best model based on validation performance  
- `confusion_matrix.png` → visual summary of class-wise predictions

### During Testing:
- Output images/videos with bounding boxes, class labels, and confidence scores  
- Example:
  ```
  detections/
   ├── img1_pred.jpg
   ├── img2_pred.jpg
  ```

Interpret the results as follows:
- **Bounding Box Color:** Represents detected class  
- **Confidence Score:** Model’s certainty about the detection  
- **Low Confidence (<0.4)** → may indicate false positives or ambiguous cases  

---

## 🧾 Notes

- Modify hyperparameters in `config.yaml` or directly in the training command for optimization.
- Ensure the dataset is correctly annotated in YOLO format (one `.txt` file per image).
- Use GPU for faster training and inference (`torch.cuda.is_available()`).

---

---

## 📁 Repository Structure
```
📦 Voltix
 ┣ 📂 examples/
 ┣ 📂 runs/
 ┣ 📜 classes.txt
 ┣ 📜 train.py
 ┣ 📜 predict.py
 ┣ 📜 visualize.py
 ┣ 📜 yolo_model.pt
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📜 data.yaml
```
---
