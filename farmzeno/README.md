# FarmZeno — Smart Farm Advisory System

AI-powered farm advisory with live weather, IMD bulletins, plant disease detection, ML crop recommendations, and user accounts with saved reports.

## Setup Instructions (VSCode)

### Step 1 — Place your model files

```
farmzeno/
└── models/
    ├── trained_model (1).keras   ← plant disease CNN model
    ├── class_names.json          ← disease class labels
    ├── best_crop_model.pkl       ← crop recommendation RandomForest
    ├── crop_scaler.pkl           ← StandardScaler
    └── crop_label_encoder.pkl    ← LabelEncoder
```

### Step 2 — Create virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app
```bash
python app.py
```

### Step 5 — Open in browser
Visit: **http://localhost:5000**

You'll be redirected to the login page. Click **Create one free** to register.

---

## New in this version — Authentication & Report Library

- **User registration and login** — each user has their own account
- **My Reports** — every advisory and disease PDF you generate can be saved to your personal library
- **Download anytime** — re-download any saved report from My Reports
- **Delete reports** — remove reports you no longer need
- **SQLite database** — user data and report metadata stored in `farmzeno.db` (auto-created on first run)
- **Secure passwords** — hashed with Werkzeug (never stored in plain text)

---

## Folder Structure

```
farmzeno/
├── app.py                    ← Flask backend (all routes + auth + logic)
├── requirements.txt
├── README.md
├── farmzeno.db               ← SQLite database (auto-created)
│
├── models/
│   ├── disease_model.py
│   ├── crop_model.py
│   └── (your model files here)
│
├── templates/
│   ├── base.html             ← navbar + user menu
│   ├── login.html            ← login page with farm background
│   ├── register.html         ← registration page
│   ├── my_reports.html       ← user report library
│   ├── home.html
│   ├── advisory.html
│   ├── disease.html
│   └── crop.html
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/logo.svg
│
├── user_reports/             ← saved PDF reports per user (auto-created)
├── uploads/                  ← temp image uploads (auto-cleared)
└── imd_cache/                ← cached IMD bulletins (auto-created)
```

## Exporting crop model files from Colab

```python
import joblib
joblib.dump(best_model,    "best_crop_model.pkl")
joblib.dump(scaler,        "crop_scaler.pkl")
joblib.dump(label_encoder, "crop_label_encoder.pkl")
```
