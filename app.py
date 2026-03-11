import gradio as gr
import pandas as pd
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'soccer_pipeline.pkl')    
ENCODER_PATH = os.path.join(BASE_DIR, 'position_encoder.pkl')  
DATA_PATH = os.path.join(BASE_DIR, 'soccer_player_performances.csv')
HISTORY_PATH = os.path.join(BASE_DIR, 'history.csv')


try:
    model_pipeline = joblib.load(MODEL_PATH)
    le_position = joblib.load(ENCODER_PATH)

    df_rec = pd.read_csv(DATA_PATH)
    unique_names = sorted(df_rec['Name'].astype(str).unique().tolist())
    unique_clubs = sorted(df_rec['Club'].astype(str).unique().tolist())
    unique_nationalities = sorted(df_rec['Nationality'].astype(str).unique().tolist())
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()


def predict_performance(name, age, club, nationality, position, appearances, assist, 
                        dist_90, goals, conceded, interception, key_passes, 
                        pass_att_90, pass_comp_90, shots_on_target, shots, 
                        tackle_att, tackle_won, shutouts):
    
    if appearances == 0:
        prediction = "-"
    else:
        input_data = pd.DataFrame([{
            'Age': age,
            'Position': position, 
            'Appearances': appearances,
            'Assist': assist,
            'Distance / 90 minutes': dist_90,
            'Goals': goals,
            'Conceded': conceded,
            'Interception': interception,
            'Key Passes': key_passes,
            'Pass Attempt / 90 minutes': pass_att_90,
            'Pass Completed / 90 minutes': pass_comp_90,
            'Shots on Target': shots_on_target,
            'Shots': shots,
            'Tackle Attempt': tackle_att,
            'Tackle Won': tackle_won,
            'Shutouts': shutouts
        }])

        epsilon = 1e-6
        input_data['Tackle_Success_Rate'] = input_data['Tackle Won'] / (input_data['Tackle Attempt'] + epsilon)
        input_data['Shot_Accuracy'] = input_data['Shots on Target'] / (input_data['Shots'] + epsilon)
        input_data['Goal_Conversion'] = input_data['Goals'] / (input_data['Shots'] + epsilon)
        input_data['Pass_Accuracy'] = input_data['Pass Completed / 90 minutes'] / (input_data['Pass Attempt / 90 minutes'] + epsilon)
        input_data['Goals_Per_App'] = input_data['Goals'] / (input_data['Appearances'] + epsilon)
        input_data['Assists_Per_App'] = input_data['Assist'] / (input_data['Appearances'] + epsilon)

        try:
            input_data['Position'] = le_position.transform(input_data['Position'])

            prediction = model_pipeline.predict(input_data)[0]
        except Exception as e:
            return "Error", f"Prediction Failed: {str(e)}"

    log_entry = pd.DataFrame([{
        'Name': name,
        'Predicted_Performance': prediction,
        'Club': club,
        'Nationality': nationality,
        'Position': position, 
        'Age': age,
        'Appearances': appearances,
        'Assist': assist,
        'Goals': goals,
        'Interception': interception,
        'Shots': shots
    }])

    write_header = not os.path.isfile(HISTORY_PATH)
    log_entry.to_csv(HISTORY_PATH, mode='a', header=write_header, index=False)

    return prediction, "Saved to history.csv"

inputs = [
    gr.Dropdown(choices=unique_names, label="Player Name", allow_custom_value=True, filterable=True),
    gr.Number(label="Age", value=25),
    gr.Dropdown(choices=unique_clubs, label="Club", allow_custom_value=True, filterable=True),
    gr.Dropdown(choices=unique_nationalities, label="Nationality", allow_custom_value=True, filterable=True),
    
    gr.Dropdown(choices=['Defender', 'Goalkeeper', 'Midfielder', 'Striker'], label="Position", value="Midfielder"),
    
    gr.Number(label="Appearances", value=0, info="If 0, Performance will be '-'"),
    gr.Number(label="Assists", value=0),
    gr.Number(label="Distance / 90 min (km)", value=0.0),
    gr.Number(label="Goals", value=0),
    gr.Number(label="Conceded (Goals Allowed)", value=0),
    gr.Number(label="Interceptions", value=0),
    gr.Number(label="Key Passes", value=0),
    gr.Number(label="Pass Attempt / 90 min", value=0.0),
    gr.Number(label="Pass Completed / 90 min", value=0.0),
    gr.Number(label="Shots on Target", value=0),
    gr.Number(label="Total Shots", value=0),
    gr.Number(label="Tackle Attempts", value=0),
    gr.Number(label="Tackles Won", value=0),
    gr.Number(label="Shutouts (Clean Sheets)", value=0),
]

outputs = [
    gr.Label(label="Predicted Performance"),
    gr.Label(label="System Status")
]

title = "Soccer Player Performance Predictor"
description = """
**Instructions:**
1. Enter the player's match statistics below.
2. Click **Submit** to generate a performance rating (Bad, Normal, Good).
3. If 'Appearances' is 0, the system will automatically assign a rating of '-'.
4. All queries are automatically saved to `history.csv`.
"""

app = gr.Interface(
    fn=predict_performance, 
    inputs=inputs, 
    outputs=outputs, 
    title=title, 
    description=description,
    theme=gr.themes.Default()
)

if __name__ == "__main__":
    app.launch(share=True)