from dash import Dash, html, dcc, Input, Output
import pandas as pd
import joblib
import numpy as np

# Load both model and scaler
model, scaler = joblib.load('selling_price.model')

# Initialize the Dash app
app = Dash()

# Layout
app.layout = html.Div(
    children=[
        # Header
        html.H1(children="Car Price Prediction", style={'textAlign': 'center'}),

        # Instructions Section
        html.Div(
            children=[
                html.H2("Instruction"),
                html.P("In order to predict car price, you need to choose Year, Mileage, Kilometer Driven, and Number of previous owners"),
                html.P("If you don't know the values, default values can help you predict."),
            ],
            style={'marginBottom': '20px'}
        ),

        # Input Fields
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Label("Year"),
                        dcc.Input(id="year", type="number", value=2023, style={'width': '100%', 'padding': '8px'}),
                    ],
                    style={'marginBottom': '15px'}
                ),
                html.Div(
                    children=[
                        html.Label("Mileage (kmpl)"),
                        dcc.Input(id="mileage", type="number", value=20, style={'width': '100%', 'padding': '8px'}),
                    ],
                    style={'marginBottom': '15px'}
                ),
                html.Div(
                    children=[
                        html.Label("Kilometer Driven (km)"),
                        dcc.Input(id="km_driven", type="number", value=14000, style={'width': '100%', 'padding': '8px'}),
                    ],
                    style={'marginBottom': '15px'}
                ),
                html.Div(
                    children=[
                        html.Label("Number of Owners"),
                        dcc.Dropdown(
                            id="owner",
                            options=[
                                {"label": "First Owner", "value": 1},
                                {"label": "Second Owner", "value": 2},
                                {"label": "Third Owner", "value": 3},
                                {"label": "Fourth Owner", "value": 4},
                            ],
                            value=1,  # Default selected value
                            style={'width': '100%', 'padding': '8px'}
                        ),
                    ],
                    style={'marginBottom': '15px'}
                ),
            ]
        ),

        # Predict Button
        html.Button("predict", id="predict-button", n_clicks=0, style={
            'width': '100%',
            'padding': '10px',
            'backgroundColor': '#007bff',
            'color': 'white',
            'border': 'none',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'fontSize': '16px'
        }),

        # Prediction Result
        html.Div(
            id="result",
            children="The predicted car price will appear here.",
            style={'marginTop': '20px', 'fontWeight': 'bold'}
        ),
    ],
    style={'maxWidth': '600px', 'margin': '0 auto', 'fontFamily': 'Arial, sans-serif'}
)

# Callback for prediction
@app.callback(
    Output("result", "children"),
    [
        Input("predict-button", "n_clicks"),
        Input("year", "value"),
        Input("mileage", "value"),
        Input("km_driven", "value"),
        Input("owner", "value"),
    ]
)
def predict_price(n_clicks, year, mileage, km_driven, owner):
    if n_clicks > 0:  # Check if the button was clicked
        try:
            if year is None or km_driven is None or mileage is None or owner is None:
                return "Please fill in all fields before predicting."

            # Create input DataFrame with correct column names
            input_data = pd.DataFrame({
                "year": [year],
                "mileage": [mileage],
                "km_driven": [km_driven],
                "owner": [owner],
            })

            # Scale input using the same scaler from training
            input_data_scaled = scaler.transform(input_data)
            
            # Debug log
            print(f"Scaled input for prediction:\n{input_data_scaled}")

            # Raw model prediction (log-price or raw price depending on training)
            raw_pred = model.predict(input_data_scaled)[0]

            # If your model was trained on log(price), convert back:
            try:
                predicted_price = np.exp(raw_pred)
            except OverflowError:
                predicted_price = raw_pred  # fallback in case it was trained on raw price

            return f"The predicted car price is ≈ {predicted_price:,.2f} Baht"
        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"An error occurred: {e}"
    return "Click the predict button to see the result."


if __name__ == "__main__":
    app.run(debug=True)