from flask import Flask
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import matplotlib
import csv

# Set matplotlib to use the 'Agg' backend to avoid GUI issues
matplotlib.use('Agg')

app = Flask(__name__)

# Load ML model (RandomForest or other trained model)
model_path = 'models/random_forest_sleep_cvd_model.pkl'
model = pickle.load(open(model_path, 'rb'))

# Define the CSV file path where form data will be stored
csv_file_path = 'data/form_data.csv'

# Route to serve the index
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/bmi1')
def bmi1():
    return render_template('bmi1.html')

# Route to serve the form
@app.route('/form')
def form():
    return render_template('form.html')

# Route to handle form submission and process data
@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        # Collect form data
        cvd_diagnosed = 1 if request.form['cvd_diagnosed'].lower() == 'yes' else 0
        age = int(request.form['age'])
        gender_str = request.form['gender']
        gender = 1 if gender_str.lower() == 'male' else 0
        sleep_hours = float(request.form['sleep_hours'])
        sleep_quality = int(request.form['sleep_quality'])
        insomnia = 1 if request.form['insomnia'] == 'yes' else 0
        sleep_apnea = 1 if request.form['sleep_apnea'] == 'yes' else 0
        snoring = 1 if request.form['snoring'] == 'yes' else 0
        daytime_sleepiness = 1 if request.form['daytime_sleepiness'] == 'yes' else 0
        hypertension = 1 if request.form['hypertension'] == 'yes' else 0
        diabetes = 1 if request.form['diabetes'] == 'yes' else 0
        smoking_status_str = request.form['smoking_status']
        smoking_status = 0 if smoking_status_str == 'never' else 1 if smoking_status_str == 'former' else 2
        physical_activity = int(request.form['physical_activity'])
        bmi = float(request.form['bmi'])
        stress_level = int(request.form['stress_level'])

        # Log the inputs for debugging
        print(f"Form Data: {cvd_diagnosed}, {age}, {gender}, {sleep_hours}, {sleep_quality}, {insomnia}, {sleep_apnea}, {snoring}, {daytime_sleepiness}, {smoking_status}, {physical_activity}, {bmi}, {hypertension}, {diabetes}, {stress_level}")

        # Prepare input for the model
        input_data = np.array([age, gender, sleep_hours, sleep_quality, insomnia, sleep_apnea, snoring, daytime_sleepiness, smoking_status, physical_activity, bmi, hypertension, diabetes, stress_level]).reshape(1, -1)

        # Make prediction using probabilities
        prediction_probs = model.predict_proba(input_data)[0]  # Get prediction probabilities
        risk_score = prediction_probs[1]  # Assuming index 1 is the probability of high risk

        # Categorize risk into Low, Medium, High
        if risk_score >= 0.7:
            risk_level = "High"
        elif 0.5 <= risk_score < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        risk_percentages = {
            "CAD": np.random.uniform(10, 50) if risk_level == "High" else np.random.uniform(5, 25),
            "Hypertension": np.random.uniform(10, 50) if hypertension == 1 else np.random.uniform(5, 15),
            "Sleep Apnea": np.random.uniform(5, 25) if sleep_apnea == 1 else np.random.uniform(1, 10),
            "Peripheral Artery Disease": np.random.uniform(10, 30) if risk_level == "High" else np.random.uniform(1, 10)
        }

        # Determine sleep chronotype based on sleep hours
        if 7.5 <= sleep_hours <= 8:
            chronotype = "Bear"
        elif 6 <= sleep_hours < 7.5:
            chronotype = "Wolf"
        elif 4.5 < sleep_hours < 6:
            chronotype = "Dolphin"
        elif 8 < sleep_hours <= 9:
            chronotype = "Lion"
        else:
            chronotype = "Owl"

        chronotype_descriptions = {
            "Bear": "Bears follow a sleep-wake pattern that aligns with the sun. They wake up naturally around sunrise and feel energized throughout the day. Most productive during the mid-morning, they tend to feel tired in the late afternoon and early evening.",
            "Wolf": "Wolves are natural night owls who feel most energetic in the late afternoon and evening. They have trouble waking up early and prefer staying up late. Their productivity peaks later in the day.",
            "Dolphin": "Dolphins are light sleepers who often struggle with insomnia. They wake up feeling unrested, and their sleep is frequently interrupted. Dolphins are sensitive to sleep environments and tend to have irregular sleep patterns.",
            "Lion": "Lions are early risers and morning people. They are most productive in the morning and tend to go to bed early. Lions feel energized in the morning but lose energy as the day progresses.",
            "Owl": "Owls are extreme night owls who thrive during late-night hours. They prefer staying up late and have difficulty waking up early. Their productivity and energy levels peak late in the night."
        }

        chronotype_info = chronotype_descriptions[chronotype]

        # Provide recommendations based on cardiovascular disease risk
        recommendations = []
        if risk_percentages['CAD'] > 25:
            recommendations.append("Reduce saturated fats and increase physical activity to lower CAD risk. Focus on consuming heart-healthy fats like those found in olive oil, nuts, seeds, and avocados. Additionally, aim for at least 150 minutes of moderate-intensity aerobic activity (such as brisk walking or cycling) per week to improve cardiovascular health.")
        
        if risk_percentages['Hypertension'] > 25:
            recommendations.append("Consider lowering your salt intake and managing stress to control hypertension. The recommended sodium intake is less than 2,300 mg per day, ideally closer to 1,500 mg. Engage in stress-reducing activities like mindfulness, yoga, or deep-breathing exercises. Regular physical activity and maintaining a healthy weight also contribute to better blood pressure control.")
    
        if risk_percentages['Sleep Apnea'] > 15:
            recommendations.append("Consult with a specialist regarding sleep apnea treatment, such as CPAP. Continuous Positive Airway Pressure (CPAP) therapy can help maintain open airways during sleep, improving breathing and reducing the risks associated with untreated sleep apnea, such as heart disease and stroke. Weight loss, avoiding alcohol before bedtime, and sleeping on your side may also improve symptoms.")
            
        if risk_percentages['Peripheral Artery Disease'] > 15:
            recommendations.append("Engage in regular physical activity and maintain a healthy diet to prevent PAD. Walking and other low-impact exercises can help improve circulation. Eating a balanced diet rich in fruits, vegetables, whole grains, and lean proteins supports arterial health. Additionally, avoid smoking, as it significantly increases the risk of PAD and other cardiovascular diseases.")
            
        else:
            recommendations.append("Maintain a balanced diet and regular physical activity to support overall cardiovascular health.")
                                   
        # Save form data to CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cvd_diagnosed, age, gender_str, sleep_hours, sleep_quality, insomnia, sleep_apnea, snoring, daytime_sleepiness, smoking_status_str, physical_activity, bmi, hypertension, diabetes, stress_level, risk_level, chronotype])

        # Generate the donut chart
        create_pie_chart(risk_percentages)

        # Render results in the new page
        return render_template(
            'result.html',
            prediction=risk_level,  # Now shows "High", "Medium", or "Low"
            risk_percentages=risk_percentages,
            recommendations=recommendations,
            chronotype=chronotype,
            chronotype_info=chronotype_info
        )

    except Exception as e:
        print(f"Error processing the form: {e}")
        return f"Error occurred during processing: {e}"


# Function to create the pie chart
def create_pie_chart(risk_percentages):
    labels = list(risk_percentages.keys())
    sizes = list(risk_percentages.values())
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']
    explode = (0.1, 0, 0, 0)

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')

    image_directory = 'static/images'
    image_path = os.path.join(image_directory, 'chart.png')

    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    plt.savefig(image_path)
    plt.close()

    return image_path

if __name__ == "__main__":
    app.run(debug=True)
