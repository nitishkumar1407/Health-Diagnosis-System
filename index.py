import numpy as np
import pandas as pd
from tkinter import Tk, StringVar, Label, CENTER, W, Button, OptionMenu, Text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt

class HealthDiagnosisSystem:
    def __init__(self):
        # Load model and data when initializing the app
        self.model, self.mlb, self.Le, self.X_test, self.Y_test = self.load_model()
        self.all_symptoms = self.mlb.classes_
        
        # Create the GUI
        self.root = Tk()
        self.root.title("Health-Diagnosis-System")
        self.create_widgets()
        
    def load_model(self):
        """Load and train the disease prediction model"""
        # Load dataset
        df = pd.read_csv(r"/Users/nitishkumar/Desktop/Health-Diagnosis-System/dataset.csv")
        
        # Combine all symptom columns into a list per row
        symptoms_cols = df.columns[1:]
        df["all_symptoms"] = df[symptoms_cols].values.tolist()
        df["all_symptoms"] = df["all_symptoms"].apply(lambda x: [symptom for symptom in x if pd.notnull(symptom)])

        # Encode symptoms using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        X = mlb.fit_transform(df["all_symptoms"])

        # Encode diseases using LabelEncoder
        Le = LabelEncoder()
        Y = Le.fit_transform(df["Disease"])
        
        # Train/test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=40)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=40)
        model.fit(X_train, Y_train)

        return model, mlb, Le, X_test, Y_test
    
    def encode_symptoms(self, symptom_list):
        """Convert list of symptoms to binary vector"""
        return [1 if symptom in symptom_list else 0 for symptom in self.all_symptoms]
    
    def predict_disease(self):
        """Handle prediction when button is clicked"""
        # Get selected symptoms
        symptoms = []
        if self.Symptom1.get() != "None":
            symptoms.append(self.Symptom1.get())
        if self.Symptom2.get() != "None":
            symptoms.append(self.Symptom2.get())
        if self.Symptom3.get() != "None":
            symptoms.append(self.Symptom3.get())
        if self.Symptom4.get() != "None":
            symptoms.append(self.Symptom4.get())
        if self.Symptom5.get() != "None":
            symptoms.append(self.Symptom5.get())
            
        if not symptoms:
            self.t3.delete(1.0, "end")
            self.t3.insert("end", "Please select at least one symptom")
            return
            
        # Convert to binary vector
        user_vector = self.encode_symptoms(symptoms)
        
        # Predict probabilities
        probabilities = self.model.predict_proba([user_vector])[0]
        disease_names = self.Le.inverse_transform(np.arange(len(probabilities)))
        
        # Get top predictions
        top_diseases = []
        for disease, prob in zip(disease_names, probabilities):
            if prob > 0.01:  # Show only if probability > 1%
                top_diseases.append(f"{disease}: {prob*100:.2f}%")
        
        # Display results
        self.t3.delete(1.0, "end")
        if top_diseases:
            result_text = "\n".join(top_diseases[:5])  # Show top 5 results
            location_info = f"\n\nLocation: {self.location.get()}" if self.location.get() != "None" else ""
            self.t3.insert("end", "Most likely diseases:\n" + result_text + location_info)

            # Plotting the top 5 predicted diseases
            self.plot_top_diseases(disease_names, probabilities)
        else:
            self.t3.insert("end", "No significant predictions found")
    
    def plot_top_diseases(self, disease_names, probabilities):
        """Plot the top 5 disease predictions using matplotlib"""
        # Sort by probabilities
        sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
        top_indices = sorted_indices[:5]  # Get top 5 predictions

        top_diseases = [disease_names[i] for i in top_indices]
        top_probs = [probabilities[i] * 100 for i in top_indices]

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.bar(top_diseases, top_probs, color='orange')
        plt.xlabel("Diseases")
        plt.ylabel("Probability (%)")
        plt.title("Top 5 Disease Prediction Probabilities")
        plt.tight_layout()
        plt.show()

    def create_widgets(self):
        """Create all GUI widgets"""
        # Initialize symptom variables
        self.Symptom1 = StringVar()
        self.Symptom1.set("None")
        self.Symptom2 = StringVar()
        self.Symptom2.set("None")
        self.Symptom3 = StringVar()
        self.Symptom3.set("None")
        self.Symptom4 = StringVar()
        self.Symptom4.set("None")
        self.Symptom5 = StringVar()
        self.Symptom5.set("None")
        self.location = StringVar()
        self.location.set("None")

        # Main title
        w2 = Label(self.root, justify=CENTER, text="Disease Prediction From Symptoms")
        w2.config(font=("Helvetica", 30))
        w2.grid(row=1, column=0, columnspan=2, padx=100)

        # Symptom labels and dropdowns
        S1Lb = Label(self.root, text="Symptom 1")
        S1Lb.config(font=("Helvetica", 15))
        S1Lb.grid(row=7, column=1, pady=10, sticky=W)

        S2Lb = Label(self.root, text="Symptom 2")
        S2Lb.config(font=("Helvetica", 15))
        S2Lb.grid(row=8, column=1, pady=10, sticky=W)

        S3Lb = Label(self.root, text="Symptom 3")
        S3Lb.config(font=("Helvetica", 15))
        S3Lb.grid(row=9, column=1, pady=10, sticky=W)

        S4Lb = Label(self.root, text="Symptom 4")
        S4Lb.config(font=("Helvetica", 15))
        S4Lb.grid(row=10, column=1, pady=10, sticky=W)

        S5Lb = Label(self.root, text="Symptom 5")
        S5Lb.config(font=("Helvetica", 15))
        S5Lb.grid(row=11, column=1, pady=10, sticky=W)

        locLb = Label(self.root, text="Location")
        locLb.config(font=("Helvetica", 15))
        locLb.grid(row=12, column=1, pady=10, sticky=W)

        # Options for dropdowns - using the symptoms from the model
        OPTIONS = sorted(self.all_symptoms)  # Use the actual symptoms from the model
        LOCATIONS = ["Patna","Bhopal","Pune","Hyderabad","New Delhi", "Mumbai", "Chennai", "Kolkata", "Bengaluru"]

        # Create dropdown menus
        S1En = OptionMenu(self.root, self.Symptom1, *OPTIONS)
        S1En.grid(row=7, column=2)

        S2En = OptionMenu(self.root, self.Symptom2, *OPTIONS)
        S2En.grid(row=8, column=2)

        S3En = OptionMenu(self.root, self.Symptom3, *OPTIONS)
        S3En.grid(row=9, column=2)

        S4En = OptionMenu(self.root, self.Symptom4, *OPTIONS)
        S4En.grid(row=10, column=2)

        S5En = OptionMenu(self.root, self.Symptom5, *OPTIONS)
        S5En.grid(row=11, column=2)

        LocEn = OptionMenu(self.root, self.location, *LOCATIONS)
        LocEn.grid(row=12, column=2)

        # Predict button
        lr = Button(self.root, text="Predict", height=2, width=20, command=self.predict_disease)
        lr.config(font=("Helvetica", 15))
        lr.grid(row=15, column=1, columnspan=2, pady=20)

        # Result display area
        NameLb = Label(self.root, text="Prediction Result:")
        NameLb.config(font=("Helvetica", 15))
        NameLb.grid(row=17, column=1, pady=10, sticky=W)

        self.t3 = Text(self.root, height=10, width=50)  # Increased height for better display
        self.t3.config(font=("Helvetica", 12))
        self.t3.grid(row=18, column=1, columnspan=2, padx=10, pady=10)

        # Display model accuracy
        accuracy = self.model.score(self.X_test, self.Y_test)
        accuracy_label = Label(self.root, text=f"Model Accuracy: {accuracy:.2%}")
        accuracy_label.config(font=("Helvetica", 12))
        accuracy_label.grid(row=19, column=1, columnspan=2, pady=10)

        self.root.mainloop()

# Run the application
HealthDiagnosisSystem()