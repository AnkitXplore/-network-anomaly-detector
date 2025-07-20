# CyberShield Pro - AI Network Security

---

**🔗 [Download All Project Files from Google Drive](https://drive.google.com/drive/folders/1mIkieZtXxDGVyi6es_Xh83zjVKoMJV7f?usp=drive_link)**

> This Google Drive folder contains all the files required to run the project, including models, datasets, and any additional resources.

---

# Network Traffic Anomaly Detection + AI Chatbot

![Network Anomaly Detection Banner](https://imgur.com/LkDpz9V.png)

<h1 align="center"> [31m [1m [4mNetwork Traffic Anomaly Detection + AI Chatbot [0m</h1>

<p align="center">
  <b> [33mReal-time anomaly detection system using Machine Learning + Streamlit + AI Chatbot [0m</b><br>
  <i>Secure your networks, visualize predictions, chat with an AI assistant, and detect attacks before they strike.</i>
</p>

---

##  [34mOverview [0m

This web application allows users to upload network traffic data (like NSL-KDD / KDDTest+), preprocess it, and make real-time predictions on whether the traffic is normal or an anomaly using a trained machine learning model.

**New!** Now includes an integrated AI Chatbot (powered by OpenRouter LLMs) to answer questions, explain results, and provide cybersecurity guidance.

---

##  [34mTech Stack [0m

| Layer      | Tools Used                                 |
| ---------- | ------------------------------------------ |
| Frontend   | Streamlit, HTML (via Streamlit components) |
| Backend    | Python, Pandas, Scikit-Learn, Joblib       |
| AI Chatbot | OpenRouter LLMs (DeepSeek, GPT-4o, etc.)   |
| Deployment | Streamlit Cloud                            |
| Model      | Random Forest Classifier                   |

---

##  [34mFeatures [0m

- Upload CSV files like `KDDTest+`
- Preprocessing + Label Encoding
- Model Predictions (Anomaly or Normal)
- Real-time, minimal, and responsive UI
- **AI Chatbot in Sidebar** (ask about results, get explanations, cybersecurity tips)
- Persistent API key (set once in code, works for all users)
- Easy to deploy with one click
- Designed for showcasing during interviews & hackathons

---

##  [34mHow to Run Locally [0m

```bash
# 1. Clone the repository
git clone https://github.com/your-username/netwok-detection-new-file.git
cd netwok\ detection\ new\ file

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Retrain the ML model if needed
python src/train_model.py

# 5. Set your OpenRouter API key (in streamlit_app/streamlit_app.py)
# Example:
# OPENROUTER_API_KEY = "sk-or-...yourkey..."

# 6. Run the Streamlit app
streamlit run streamlit_app/streamlit_app.py
```

---

##  [34mProject Structure [0m

```
netwok detection new file/
├── app.py
├── cleaned_header.txt
├── data/
│   ├── data/
│   │   └── raw/
│   │       ├── KDDTest+.csv
│   │       └── KDDTrain+.csv
│   └── processed/
│       └── cleaned_KDDTrain.csv
├── debug_model.py
├── models/
│   └── models/
│       └── Network_Anomility.joblib
├── notebooks/
│   ├── Network_Anomility.joblib
│   ├── preprocessing.ipynb
│   └── train_model.ipynb
├── requirements.txt
├── src/
│   ├── predict_live.py
│   ├── preprocessing.ipynb
│   └── train_model.py
├── streamlit_app/
│   └── streamlit_app.py
└── test_html.py
```

---

##  [34mModel Files [0m

- The trained model (`Network_Anomility.joblib`) is stored in the `models/` directory.
- If the model file is not present, run the training script as shown above.
- **Note:** Large model files and datasets are excluded from version control via `.gitignore`. If you want to share them, use a cloud link or Git LFS.

---

##  [34mLicense [0m

This project is licensed under the MIT License. See `LICENSE` for details.

---

> Built with ❤️ by [Ankit Tiwari](https://www.linkedin.com/in/ankit-tiwari-198772240)
