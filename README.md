# Health-Diagnosis-System  
## AI-based Health Diagnosis System using Python and Machine Learning

📌 **Project Description:**  
This Health Diagnosis System is a desktop application developed using Python. It helps users identify possible diseases based on the symptoms they input. The system uses a machine learning model trained on a dataset of diseases and associated symptoms. Once the user enters 2–5 symptoms, the system predicts the most likely diseases and displays the results along with confidence percentages.

The GUI (Graphical User Interface) is built using **Tkinter**, and the results are visualized using **Matplotlib**. This tool can assist in early-stage diagnosis or awareness, helping users seek timely medical attention.

---

## 🛠️ Technologies Used:
- Python  
- Tkinter (GUI)  
- Pandas  
- NumPy  
- Scikit-learn (ML Model)  
- Matplotlib (for data visualization)  
- CSV (Dataset)

---

## 💻 How to Run the Project:

### 1. Clone the Repository:
```bash
git clone https://github.com/nitishkumar1407/Health-Diagnosis-System.git
cd Health-Diagnosis-System
```

### 2. Install Required Packages:
Make sure you have Python installed, then install the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib
```


---

## 🧠 How it Works:
1. **Dataset**: A CSV file containing diseases and associated symptoms is used to train the model.  
2. **Model Training**: The model uses supervised learning to map symptoms to diseases.  
3. **User Input**: The user selects symptoms through the GUI.  
4. **Prediction**: The trained ML model predicts possible diseases with confidence scores.  
5. **Output**: The predicted diseases are displayed in the GUI, and a bar graph of confidence levels is shown using Matplotlib.

---

## 📁 Folder Structure:
```text
Health-Diagnosis-System/
├── main.py                  # Entry point with GUI
├── model.py                 # ML model logic
├── train_model.py          # Code to train and save the model
├── data.csv                # Dataset used for training
├── model.pkl               # Trained model file
├── README.md               # Project documentation
```

---


## 🙋‍♀️ Contributors
This is a collaborative group project by passionate undergraduates from LNCT College, addressing real-world energy forecasting challenges using ML.

- Haripriya Mahajan (Backend & Model Development)
-- GitHub: https://github.com/Haripriya-Mahajan

- Aditya Garg (Backend & Model Development)
-- GitHub: https://github.com/Adiigarg07

- Nitish Kumar (Frontend with Streamlit)
-- GitHub: https://github.com/nitishkumar1407

- Anjali Patel (Frontend with Streamlit)
-- GitHub: https://github.com/anjalip2623

---

## 📌 Note:
> This project is intended for educational purposes and not for real medical diagnosis. Always consult a healthcare professional for accurate diagnosis and treatment.
