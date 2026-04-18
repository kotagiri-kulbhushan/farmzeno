# 🌱 FarmZeno — AI Smart Crop Advisory System

FarmZeno is an intelligent agriculture assistant that helps farmers make **data-driven decisions** using **Machine Learning, Weather APIs, and AI-powered advisory systems**.

---

## 🚀 Key Features

### 🌦 Real-Time Weather Advisory

* Live weather data using OpenWeather API
* 24-hour forecast insights
* Smart recommendations based on weather conditions

---

### 🌾 Crop Recommendation System

* Machine Learning (Random Forest)
* Inputs: Temperature, Humidity, pH, Rainfall
* Outputs:

  * Best crop recommendation
  * Top 5 alternative crops

---

### 🌿 Plant Disease Detection

* Deep Learning (TensorFlow CNN)
* Image-based disease classification
* Confidence score + Top predictions

---

### 📄 Smart Advisory Report

* Generates clean, one-page professional PDF reports
* Includes:

  * Weather summary
  * Crop suggestions
  * Farming recommendations

---

### 🧠 AI Advisory Engine

* Uses IMD bulletins + FAISS similarity search
* Provides:

  * Weather warnings
  * Pest/disease alerts
  * Irrigation advice
  * Fertilizer & spraying guidance

---

# 📊 Model Performance

## 📌 Accuracy Summary (Best Models)

| Module              | Model            | Accuracy                |
| ------------------- | ---------------- | ----------------------- |
| Crop Recommendation | Random Forest    | **95.64%**              |
| Disease Detection   | CNN (TensorFlow) | **95.35% (Validation)** |

---

## 🌾 Crop Recommendation Model

![Crop Model](accuracy%20images/crop%20model.png)

---

## 🌿 Plant Disease Detection Model

![Validation Accuracy](accuracy%20images/training_accuracy.jpeg)

---

# 🖥 Application Screenshots

## 🌦 Advisory System

![Advisory](https://raw.githubusercontent.com/kotagiri-kulbhushan/Farmzeno/main/UI/FarmZeno-%E2%80%94-Crop-Advisory.png)

## 🌾 Crop Recommendation

![Crop](https://raw.githubusercontent.com/kotagiri-kulbhushan/Farmzeno/main/UI/FarmZeno-%E2%80%94-Crop-Recommendations.png)

## 🌿 Disease Detection

![Disease](https://raw.githubusercontent.com/kotagiri-kulbhushan/Farmzeno/main/UI/FarmZeno-%E2%80%94-Plant-Disease-Detection.png)

## 🌤 Weather Dashboard

![Weather](https://raw.githubusercontent.com/kotagiri-kulbhushan/Farmzeno/main/UI/FarmZeno-%E2%80%94-Weather-Dashboard.png)

## 🔐 Login Page

![Login](https://raw.githubusercontent.com/kotagiri-kulbhushan/Farmzeno/main/UI/Login-%E2%80%94-FarmZeno.png)

## 📂 Reports Page

![Reports](https://raw.githubusercontent.com/kotagiri-kulbhushan/Farmzeno/main/UI/My-Reports-%E2%80%94-FarmZeno.png)

---

# 📄 Sample Generated Reports

👉 [Download Advisory Report](Sample%20reports/FarmZeno_Advisory_phagwara_20260407.pdf)
👉 [Download Disease Report](Sample%20reports/FarmZeno_Disease_Report.pdf)

---

# 📦 Model Files (Download Required)

⚠️ Model files are not included in the repo due to size limits.

👉 Download from Google Drive:
https://drive.google.com/drive/folders/177SorOMPPzBFFraebX9vlgQJwnOFLjBT?usp=sharing

Place them inside:

```
farmzeno (project folder)/models/
```

Required files:

* `trained_model (1).keras`
* `class_names.json`
* `best_crop_model.pkl`
* `crop_scaler.pkl`
* `crop_label_encoder.pkl`

---

# ⚙️ Setup Instructions

## 1️⃣ Clone Repository

```
git clone https://github.com/kotagiri-kulbhushan/Farmzeno.git
cd Farmzeno
```

---

## 2️⃣ Create Virtual Environment

### Windows

```
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```
python -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

## 4️⃣ Add Environment Variable

Create `.env` file:

```
OPENWEATHER_API_KEY=your_api_key
```

---

## 5️⃣ Run Application

```
python app.py
```

Open:

```
http://localhost:5000
```

---

# 🔐 Authentication & Report Library

* User registration & login system
* Each user has a **personal dashboard**
* Save generated reports
* Download anytime
* Delete unwanted reports
* Passwords stored securely (hashed using Werkzeug)
* SQLite database (`farmzeno.db`) auto-created

---

# 📂 Project Structure

```
farmzeno (project folder)/
├── app.py
├── requirements.txt
├── README.md
├── farmzeno.db
│
├── models/
│   ├── disease_model.py
│   ├── crop_model.py
│   └── (downloaded model files)
│
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── my_reports.html
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
├── user_reports/   (auto-generated)
├── uploads/        (auto-generated)
└── imd_cache/      (auto-generated)
```

---

# 🚀 Future Enhancements

* 📱 Mobile application
* 🌍 Multi-language farmer support
* 🛰 Satellite data integration
* 🌡 IoT sensor integration

---

# 👨‍💻 Author

**Kul Bhushan Kotagiri**

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub!
