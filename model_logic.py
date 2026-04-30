import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator


# =========================
# Load data
# =========================
symptoms_df = pd.read_csv("disease_symptoms.csv")
info_df = pd.read_csv("disease_info_clean.csv")
doctor_df = pd.read_csv("disease_doctor.csv")

symptoms_df.columns = symptoms_df.columns.str.strip().str.lower()
info_df.columns = info_df.columns.str.strip().str.lower()
doctor_df.columns = doctor_df.columns.str.strip().str.lower()

symptoms_df["symptom"] = symptoms_df["symptom"].astype(str).str.strip()
symptoms_df["word_count"] = symptoms_df["symptom"].apply(lambda x: len(x.split()))

clean_df = symptoms_df[
    (symptoms_df["word_count"] >= 1) &
    (symptoms_df["word_count"] <= 6)
].drop_duplicates().copy()

disease_counts = clean_df["disease"].value_counts()
top_diseases = disease_counts.head(145).index
filtered_df = clean_df[clean_df["disease"].isin(top_diseases)].copy()


# =========================
# Manual symptom enhancement
# =========================
extra_rows = [
    {"disease": "Gastroesophageal reflux disease (GERD)", "symptom": "Heartburn"},
    {"disease": "Gastroesophageal reflux disease (GERD)", "symptom": "Acid reflux"},
    {"disease": "Gastroesophageal reflux disease (GERD)", "symptom": "Burning chest after eating"},

    {"disease": "Gastritis", "symptom": "Burning stomach pain"},
    {"disease": "Gastritis", "symptom": "Nausea"},
    {"disease": "Gastritis", "symptom": "Fullness after eating"},

    {"disease": "Peptic ulcer", "symptom": "Burning stomach pain"},
    {"disease": "Peptic ulcer", "symptom": "Heartburn"},
    {"disease": "Peptic ulcer", "symptom": "Bloating"},

    {"disease": "Common cold", "symptom": "Runny nose"},
    {"disease": "Common cold", "symptom": "Sneezing"},
    {"disease": "Common cold", "symptom": "Sore throat"},
    {"disease": "Common cold", "symptom": "Cough"},

    {"disease": "Influenza (flu)", "symptom": "Fever"},
    {"disease": "Influenza (flu)", "symptom": "Cough"},
    {"disease": "Influenza (flu)", "symptom": "Fever and cough"},
    {"disease": "Influenza (flu)", "symptom": "Body aches"},

    {"disease": "Urinary tract infection (UTI)", "symptom": "Burning urination"},
    {"disease": "Urinary tract infection (UTI)", "symptom": "Frequent urination"},
    {"disease": "Urinary tract infection (UTI)", "symptom": "Pelvic pain"},

    {"disease": "Migraine", "symptom": "Severe headache"},
    {"disease": "Migraine", "symptom": "Headache with nausea"},
    {"disease": "Migraine", "symptom": "Nausea"},
    {"disease": "Migraine", "symptom": "Sensitivity to light"},

    {"disease": "Asthma", "symptom": "Shortness of breath"},
    {"disease": "Asthma", "symptom": "Wheezing"},
    {"disease": "Asthma", "symptom": "Cough"},
]

filtered_df = pd.concat(
    [filtered_df[["disease", "symptom"]], pd.DataFrame(extra_rows)],
    ignore_index=True
).drop_duplicates().copy()


# =========================
# Embedding model
# =========================
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
symptom_texts = filtered_df["symptom"].tolist()
symptom_embeddings = embed_model.encode(symptom_texts)


# =========================
# Dictionaries
# =========================
arabic_map = {
    "بطني بتولع": "heartburn acid reflux burning chest after eating",
    "بطني بتحرق": "heartburn acid reflux burning chest",
    "حموضة": "heartburn acid reflux",
    "حرقان في المعدة": "heartburn acid reflux burning stomach",
    "حرقة": "heartburn",
    "بعد الاكل": "after eating",
    "بعد الأكل": "after eating",
    "مغص": "abdominal pain",
    "وجع بطن": "abdominal pain",
    "ألم بطن": "abdominal pain",
    "غثيان": "nausea",
    "ترجيع": "vomiting",
    "قيء": "vomiting",
    "نفخة": "bloating",

    "كحة": "cough",
    "كحه": "cough",
    "سعال": "cough",
    "رشح": "runny nose",
    "عطس": "sneezing",
    "احتقان": "nasal congestion",
    "التهاب الحلق": "sore throat",
    "وجع حلق": "sore throat",
    "كحه وسخونيه": "cough fever flu",
    "كحة وسخونية": "cough fever flu",
    "رشح وكحه والتهاب حلق": "runny nose cough sore throat common cold",

    "ضيق تنفس": "shortness of breath asthma",
    "صفير": "wheezing asthma",
    "عندي ضيق تنفس وصفير": "asthma shortness of breath wheezing",

    "سخونية": "fever",
    "سخونيه": "fever",
    "حرارة": "fever",
    "حراره": "fever",
    "تعب": "fatigue",
    "إرهاق": "fatigue",
    "ارهاق": "fatigue",
    "وجع جسم": "body aches",

    "صداع": "headache migraine",
    "صداع شديد": "migraine severe headache",
    "صداع شديد مع غثيان": "migraine severe headache nausea sensitivity to light",
    "صداع نصفي": "migraine headache nausea",
    "دوخة": "dizziness",
    "دوخه": "dizziness",

    "حرقان في البول": "burning urination",
    "كثرة التبول": "frequent urination",
    "كتر التبول": "frequent urination",

    "ألم وحرقان في المعدة": "burning stomach pain nausea gastritis",
    "وجع معدة وحرقان": "stomach pain burning peptic ulcer gastritis",

    "عطس ورشح وحكة": "sneezing runny nose itching allergy",
}

disease_alias = {
    "Dust mite allergy": "Allergies",
    "Pet allergy": "Allergies",
    "Penicillin allergy": "Drug allergy",
    "H1N1 flu (swine flu)": "Influenza (flu)",
    "Coronavirus disease 2019 (COVID-19)": "Common cold"
}

disease_map = {
    "Common cold": "نزلات البرد",
    "Influenza (flu)": "الإنفلونزا",
    "Migraine": "الصداع النصفي",
    "Asthma": "الربو",
    "Gastroesophageal reflux disease (GERD)": "ارتجاع المريء",
    "Peptic ulcer": "قرحة المعدة",
    "Gastritis": "التهاب المعدة",
    "Urinary tract infection (UTI)": "التهاب المسالك البولية",
    "Allergies": "الحساسية",
    "Acute sinusitis": "التهاب الجيوب الأنفية الحاد",
    "Bronchitis": "التهاب الشعب الهوائية",
    "Gastroenterologist": "طبيب جهاز هضمي",
    "Neurologist": "طبيب مخ وأعصاب",
    "Pulmonologist": "طبيب صدر",
    "Urologist": "طبيب مسالك بولية",
    "Allergist": "طبيب حساسية",
    "ENT Specialist": "طبيب أنف وأذن وحنجرة",
    "General Physician": "طبيب عام"
}

doctor_map = {
    "General Physician": "طبيب عام",
    "Gastroenterologist": "طبيب جهاز هضمي",
    "Neurologist": "طبيب مخ وأعصاب",
    "Pulmonologist": "طبيب صدر",
    "Urologist": "طبيب مسالك بولية",
    "Allergist": "طبيب حساسية",
    "ENT Specialist": "طبيب أنف وأذن وحنجرة"
}


# =========================
# Helper functions
# =========================
def normalize_arabic(user_input: str) -> str:
    text = str(user_input).strip().lower()
    extra = []

    for k, v in arabic_map.items():
        if k in text:
            extra.append(v)

    return text + " " + " ".join(extra) if extra else text


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in str(text) if '\u0600' <= c <= '\u06FF')
    return "ar" if arabic_chars > 0 else "en"


def translate_to_ar(text: str) -> str:
    try:
        if not text or str(text).strip() == "":
            return str(text)
        return GoogleTranslator(source="auto", target="ar").translate(str(text))
    except Exception:
        return str(text)


def translate_disease(name: str) -> str:
    return disease_map.get(name, translate_to_ar(name))


def translate_doctor(doc: str) -> str:
    return doctor_map.get(doc, translate_to_ar(doc))


def shorten_text(text: str, max_sentences: int = 2) -> str:
    text = str(text).strip()
    parts = [p.strip() for p in text.split(".") if p.strip()]
    return ". ".join(parts[:max_sentences]) + "." if parts else text


# =========================
# Prediction
# =========================
def predict_disease(user_input: str, top_k: int = 20):
    query_text = normalize_arabic(user_input)
    query_vec = embed_model.encode([query_text])

    similarities = np.dot(symptom_embeddings, query_vec.T).reshape(-1)
    top_idx = similarities.argsort()[-top_k:][::-1]

    disease_scores = {}
    query_lower = query_text.lower()

    for i in top_idx:
        disease = filtered_df.iloc[i]["disease"]
        symptom = filtered_df.iloc[i]["symptom"]
        score = float(similarities[i])

        disease = disease_alias.get(disease, disease)

        if disease not in disease_scores:
            disease_scores[disease] = {"score": 0.0, "matched_symptoms": []}

        disease_scores[disease]["score"] += score
        disease_scores[disease]["matched_symptoms"].append(symptom)

        sym_lower = str(symptom).lower()

        # basic boosts
        if "heartburn" in sym_lower and "heartburn" in query_lower:
            disease_scores[disease]["score"] += 8
        if "acid reflux" in sym_lower and "acid reflux" in query_lower:
            disease_scores[disease]["score"] += 8
        if "cough" in sym_lower and "cough" in query_lower:
            disease_scores[disease]["score"] += 6
        if "fever" in sym_lower and "fever" in query_lower:
            disease_scores[disease]["score"] += 6
        if "headache" in sym_lower and "headache" in query_lower:
            disease_scores[disease]["score"] += 8
        if "nausea" in sym_lower and "nausea" in query_lower:
            disease_scores[disease]["score"] += 6
        if "shortness of breath" in sym_lower and "shortness of breath" in query_lower:
            disease_scores[disease]["score"] += 8
        if "wheezing" in sym_lower and "wheezing" in query_lower:
            disease_scores[disease]["score"] += 8
        if "burning urination" in sym_lower and "burning urination" in query_lower:
            disease_scores[disease]["score"] += 8
        if "frequent urination" in sym_lower and "frequent urination" in query_lower:
            disease_scores[disease]["score"] += 8

        # digestive anchors
        if disease == "Gastroesophageal reflux disease (GERD)" and (
            "heartburn" in query_lower or "acid reflux" in query_lower or "after eating" in query_lower
        ):
            disease_scores[disease]["score"] += 20

        if disease == "Gastritis" and (
            "burning stomach" in query_lower or "stomach pain" in query_lower or "nausea" in query_lower
        ):
            disease_scores[disease]["score"] += 12

        if disease == "Peptic ulcer" and (
            "burning" in query_lower or "stomach pain" in query_lower
        ):
            disease_scores[disease]["score"] += 12

        # migraine anchors
        if disease == "Migraine":
            if "headache" in query_lower:
                disease_scores[disease]["score"] += 15
            if "nausea" in query_lower:
                disease_scores[disease]["score"] += 10
            if "migraine" in query_lower:
                disease_scores[disease]["score"] += 20
            if "headache" in query_lower and "nausea" in query_lower:
                disease_scores[disease]["score"] += 25

        # respiratory anchors
        if disease in ["Influenza (flu)", "Common cold", "Bronchitis", "COVID-19"]:
            if "cough" in query_lower and "fever" in query_lower:
                disease_scores[disease]["score"] += 15

        if disease == "Asthma":
            if "shortness of breath" in query_lower:
                disease_scores[disease]["score"] += 18
            if "wheezing" in query_lower:
                disease_scores[disease]["score"] += 18
            if "asthma" in query_lower:
                disease_scores[disease]["score"] += 15
            if "shortness of breath" in query_lower and "wheezing" in query_lower:
                disease_scores[disease]["score"] += 25

        # urinary anchor
        if disease == "Urinary tract infection (UTI)" and (
            "burning urination" in query_lower and "frequent urination" in query_lower
        ):
            disease_scores[disease]["score"] += 25

        # penalties
        if disease in [
            "Stomach cancer", "Soft palate cancer", "Wilms' tumor",
            "Waldenstrom macroglobulinemia", "Valley fever",
            "Toxic hepatitis", "Drug allergy", "Penicillin allergy",
            "Dengue fever", "Diabetic coma", "Jellyfish stings",
            "Chagas disease", "Polio", "Listeria infection",
            "Granulomatosis with polyangiitis", "Glioma",
            "Gangrene", "Mesothelioma"
        ]:
            disease_scores[disease]["score"] -= 20

    ranked = sorted(disease_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    # final overrides
    if "headache" in query_lower and "nausea" in query_lower:
        return [("Migraine", {"score": 999, "matched_symptoms": []})]

    if "shortness of breath" in query_lower and "wheezing" in query_lower:
        return [("Asthma", {"score": 999, "matched_symptoms": []})]

    if "runny nose" in query_lower and "sore throat" in query_lower:
        return [("Common cold", {"score": 999, "matched_symptoms": []})]

    return ranked[:1]


# =========================
# Retrieval
# =========================
def get_disease_info(disease: str):
    row = info_df[info_df["disease"] == disease]
    if len(row) == 0:
        return None

    row = row.iloc[0]
    return {
        "overview": row["overview"],
        "treatment": row["treatment"],
        "when": row["when_to_see_doctor"]
    }


def get_doctor(disease: str) -> str:
    row = doctor_df[doctor_df["disease"] == disease]

    if len(row) > 0:
        row = row.iloc[0]
        if "doctor_specialty" in doctor_df.columns:
            return row["doctor_specialty"]
        if "doctor" in doctor_df.columns:
            return row["doctor"]
        if "specialty" in doctor_df.columns:
            return row["specialty"]

    row = doctor_df[
        doctor_df["disease"].astype(str).str.contains(str(disease), case=False, na=False)
    ]
    if len(row) > 0:
        row = row.iloc[0]
        if "doctor_specialty" in doctor_df.columns:
            return row["doctor_specialty"]
        if "doctor" in doctor_df.columns:
            return row["doctor"]
        if "specialty" in doctor_df.columns:
            return row["specialty"]

    return "General Physician"


# =========================
# Text response
# =========================
def chatbot(user_input: str) -> str:
    predictions = predict_disease(user_input)

    if len(predictions) == 0:
        return "لم أتمكن من تحديد المرض بشكل مبدئي."

    best_disease = predictions[0][0]
    info = get_disease_info(best_disease)
    doctor = get_doctor(best_disease)

    response = "🩺 التشخيص المبدئي:\n\n"
    response += f"🎯 {translate_disease(best_disease)}\n\n"
    response += "----------------\n\n"

    if info:
        overview = shorten_text(translate_to_ar(info["overview"]), 2)
        treatment = shorten_text(translate_to_ar(info["treatment"]), 2)
        when = shorten_text(translate_to_ar(info["when"]), 2)

        response += f"📌 شرح مختصر:\n{overview}\n\n"
        response += f"💊 لتقليل الأعراض:\n{treatment}\n\n"
        response += f"🚨 متى يجب زيارة الطبيب؟\n{when}\n\n"

    response += f"👨‍⚕️ التخصص المناسب:\n{translate_doctor(doctor)}\n\n"
    response += "⚠️ هذا تشخيص مبدئي وليس بديلاً عن الطبيب."

    return response


# =========================
# JSON response for API
# =========================
def chatbot_json(user_input: str):
    lang = detect_language(user_input)
    predictions = predict_disease(user_input)

    if len(predictions) == 0:
        return {
            "success": False,
            "message": "Could not determine disease" if lang == "en" else "لم أتمكن من تحديد المرض"
        }

    best_disease = predictions[0][0]
    info = get_disease_info(best_disease)
    doctor = get_doctor(best_disease)

    if lang == "ar":
        return {
            "success": True,
            "predicted_disease": translate_disease(best_disease),
            "doctor": translate_doctor(doctor),
            "overview": shorten_text(translate_to_ar(info["overview"]), 2) if info else "",
            "treatment": shorten_text(translate_to_ar(info["treatment"]), 2) if info else "",
            "when_to_see_doctor": shorten_text(translate_to_ar(info["when"]), 2) if info else ""
        }

    return {
        "success": True,
        "predicted_disease": best_disease,
        "doctor": doctor,
        "overview": shorten_text(info["overview"], 2) if info else "",
        "treatment": shorten_text(info["treatment"], 2) if info else "",
        "when_to_see_doctor": shorten_text(info["when"], 2) if info else ""
    }