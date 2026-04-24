#  Semantic Communication for Industrial Defect Detection

This project implements an AI-based semantic communication system for industrial defect detection. Instead of transmitting full images, the system transmits semantic information (defect class + confidence), significantly reducing communication cost.

---

##  Features

- Defect detection using ResNet-18
- Semantic encoding using structured JSON
- Secure transmission via compression + encryption (Fernet)
- Decoder with shared knowledge base
- Evaluation with compression and noise analysis
- Works across NEU and MVTec datasets

---

##  Project Structure

Industrial_Defect_Detection/

REAL-IAD/  
│  
├── AddNoise.py  
├── CompareFileSizes.py  
├── SemanticEncoder.py  
├── SemanticDecoder.py  
├── decode.py  
├── keygen.py  
├── realiad_train.py  
├── realiad_pipeline.py  
├── realiad_loader.py  
├── realiad_eval.py  

mvtec_scripts/  
│  
├── helper_pipeline.py  
├── helper_train.py  
├── mvtec_train_full.py  

neu_scripts/  
│  
├── neu_train.py  
├── neu_test.py  
├── neu_evaluation.py  

README.md  

---

##  Setup

### 1. Clone the repository

```bash
git clone https://github.com/Arshbir1/Industrial_Defect_Detection.git
cd Industrial_Defect_Detection
```

---

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib cryptography jsonschema
```

---

##  Dataset Setup

### NEU Dataset

Place inside:

```
NEU-DET/
```

Structure:

```
NEU-DET/train/images/<class_name>/*.jpg
```

---

### MVTec Dataset

Place inside:

```
MVTEC-AD/
```

Structure:

```
MVTEC-AD/<category>/
   ├── train/good/
   ├── test/<defect_type>/
```

---

##  Training

###  NEU

```bash
python neu_scripts/neu_train.py
```

---

###  MVTec

```bash
python mvtec_scripts/mvtec_train_full.py
```

---

###  REAL-IAD Pipeline (Main System)

```bash
python REAL-IAD/realiad_train.py
```

---

##  Testing + Semantic Encoding

### NEU

```bash
python neu_scripts/neu_test.py
```

---

### MVTec

```bash
python mvtec_scripts/helper_pipeline.py
```

---

### REAL-IAD Full Pipeline

```bash
python REAL-IAD/realiad_pipeline.py
```

Outputs:

```
secure_output/
mvtec_output/
```

These contain compressed and encrypted semantic messages.

---

##  Semantic Communication Pipeline

Image  
→ ResNet Model  
→ {class, confidence}  
→ Semantic Encoder (JSON)  
→ Compression (zlib)  
→ Encryption (Fernet)  
→ Transmission  
→ Decoder → Interpretation  

---

##  Evaluation

### NEU

```bash
python neu_scripts/neu_evaluation.py
```

---

### REAL-IAD Evaluation

```bash
python REAL-IAD/realiad_eval.py
```

---

##  Encryption Key Setup

Generate key:

```bash
python REAL-IAD/keygen.py
```

Ensure the same key is used in:

- SemanticEncoder.py  
- SemanticDecoder.py  

---

##  Model Files (IMPORTANT)

Trained model files are not included due to GitHub size limits (25MB).

### To generate models:

Run:

```bash
python neu_scripts/neu_train.py
python mvtec_scripts/mvtec_train_full.py
python REAL-IAD/realiad_train.py
```

---

##  Key Results

- ~65× reduction in transmission size  
- Lossless semantic decoding under no noise  
- Robust to semantic noise  
- Sensitive to channel noise (encryption overhead)  
- Generalizes across datasets  

---

##  Concept Highlights

- Semantic Communication  
- Shared Knowledge Base  
- Compression vs Security Trade-off  
- Cross-dataset Generalization  

---

##  Authors

- Arshbir Singh Dang  
- Siddharth Goswami

---

##  License

This project is for academic purposes.
