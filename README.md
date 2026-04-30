# 🧠 Disease Prediction AI

AI-powered system for predicting diseases based on user symptoms.
Deployed as a REST API and ready to integrate with mobile apps (Flutter - Wateen).

---

## 🚀 Live Demo

👉 https://mennaelzyat-wateen-diseasepredication.hf.space/docs

---

## ✨ Features

* 🧠 Predict diseases from user symptoms
* ⚡ Fast REST API using Flask
* 🌍 Deployed on Hugging Face Spaces
* 🔌 Ready for mobile integration (Flutter)
* 📊 Returns disease + confidence score

---

## 🛠️ Technologies Used

* Python
* Flask
* Pandas & NumPy
* Machine Learning Model
* Hugging Face Spaces

---

## 📡 API Usage

### 🔹 POST /predict

### Request:

```json
{
  "symptoms": ["fever", "headache", "fatigue"]
}
```

### Response:

```json
{
  "disease": "Flu",
  "confidence": 0.91
}
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
```

ثم افتحي:

```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
disease-prediction-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── data/

```

---

## ⚠️ Notes

* Large model files are not included in this repository
* The deployed version is available on Hugging Face
* This project is for educational purposes only

---

## 🧠 Part of

**Wateen Healthcare System**

Used for:

* AI Diagnosis
* Symptom-based prediction
* Smart healthcare assistance

---

r
