# Web Ad vs Non-Ad Image Classifier (Demo)

This project demonstrates a binary image classification system that distinguishes **web advertisements from non-advertisements**.

It is based on work developed for a Master’s thesis and includes:

  - A pretrained CNN model (trained offline)
  - A demo evaluation pipeline for inference and visualization
  - A small demo dataset for qualitative testing

The repository is intended for demonstration and reproducibility, not for retraining the full model.

## Project Structure

```text
web-ad-classifier-demo/
├── src/
│   └── demo_eval.py            # Demo evaluation & inference script
├── models/
│   └── Pretrained_MobileNet_ad_class_with_Aug.h5
├── data/
│   └── demo/                   # Small demo dataset (ads / regular)
├── requirements.txt
└── README.md
```

## Requirements
tensorflow==2.15.* \
numpy>=1.23,<2.0 \
pillow>=9.5 \
imagehash>=4.3 \
pandas>=1.5 \
matplotlib>=3.7 \
jupyter>=1.0 \
scikit-learn>=1.3


## How to Run
**1. Create a virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Demo**
```bash
python src/demo_eval.py \
  --model-path models/Pretrained_MobileNet_ad_class_with_Aug.h5 \
  --demo-data-dir data/demo
```


Notes:
- This repository does not include:
  - Training code
  - Full datasets
  - Hyperparameter search logic
- Results shown on the demo dataset are illustrative only
- The project is intended for educational and demonstration purposes
