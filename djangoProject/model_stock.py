from flask import Flask, request, render_template, jsonify

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import base64
import json
app = Flask(__name__)


@app.route("/")
def index():
    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:

    # In[2]:

    apple = yf.Ticker("AAPL")
    apple = apple.history(period="max")
    apple

    # In[3]:

    del apple["Dividends"]
    del apple["Stock Splits"]

    # In[4]:

    apple

    # In[5]:

    apple = apple.loc["2000-01-01":].copy()
    apple

    # In[6]:

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device

    # In[7]:

    apple = apple[["Close"]]
    apple

    # In[8]:

    from copy import deepcopy as dc

    def prepare_dataframe_for_ltsm(df, n_steps):
        df = dc(df)
        for i in range(1, n_steps + 1):
            df[f"Close(t-{i})"] = df["Close"].shift(i)

        df.dropna(inplace=True)
        return df

    # In[9]:

    lookback = 7
    shifted_df_apple = prepare_dataframe_for_ltsm(apple, 7)
    shifted_df_apple

    # In[10]:

    shifted_df_apple_np = shifted_df_apple.to_numpy()
    shifted_df_apple_np

    # In[11]:

    shifted_df_apple_np.shape

    # In[12]:

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_apple_np = scaler.fit_transform(shifted_df_apple_np)
    shifted_df_apple_np

    # In[13]:

    # the last 7 days(input)
    X = shifted_df_apple_np[:, 1:]
    # the close price(output)
    y = shifted_df_apple_np[:, 0]

    X.shape, y.shape

    # In[14]:

    # t-7 to t-1 so model will get the updatest information to learn from
    X = dc(np.flip(X, axis=1))
    X

    # In[2256]:

    split_index = int(len(X) * 0.85)
    split_index

    # In[2257]:

    # 85% train, 15% test
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # In[2258]:

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # In[2259]:

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # In[2260]:

    from torch.utils.data import Dataset

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    # In[2261]:

    from torch.utils.data import DataLoader

    # provide an iterable over the give dataset. It is a convenient way to automatically manage batches of data from your dataset during the
    # training or evaluation for your machine learning model
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # In[2262]:

    for _, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        break

    # In[2686]:

    class LSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_stacked_layers, dropout_prob):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_stacked_layers = num_stacked_layers

            # Use CuDNNLSTM if available, otherwise fallback to LSTM
            try:
                from torch.nn import CuDNNLSTM
                LSTMBase = CuDNNLSTM
            except ImportError:
                LSTMBase = nn.LSTM

            # Define the LSTM layer with the 'dropout' parameter for dropout between LSTM layers
            self.lstm = LSTMBase(
                input_size,
                hidden_size,
                num_stacked_layers,
                batch_first=True,
                dropout=dropout_prob if num_stacked_layers > 1 else 0.0,
            )

            # Define a dropout layer to be applied to the output of the last LSTM layer
            self.dropout = nn.Dropout(dropout_prob)

            # Define a fully connected layer for the output
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x, h0=None, c0=None):
            batch_size = x.size(0)

            # Initialize the hidden and cell states to zeros
            if h0 is None or c0 is None:
                h0 = torch.zeros(
                    self.num_stacked_layers, batch_size, self.hidden_size
                ).to(x.device)
                c0 = torch.zeros(
                    self.num_stacked_layers, batch_size, self.hidden_size
                ).to(x.device)

            # Get the output from the LSTM layers
            out, _ = self.lstm(x, (h0, c0))

            # Apply dropout to the output of the last LSTM layer
            out = self.dropout(out[:, -1, :])

            # Get the final output from the fully connected layer
            out = self.fc(out)

            return out

    # Example usage:
    # Instantiate the LSTM model with dropout
    dropout_probability = 0.08  # For example, a 5'% dropout rate
    apple_model = LSTM(
        input_size=1,
        hidden_size=5,
        num_stacked_layers=1,
        dropout_prob=dropout_probability,
    )

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apple_model.to(device)

    # In[2687]:

    def train_one_epoch(model):
        # Set the model to training mode. This is important for models with specific layers that have
        # different behavior during training vs testing (like dropout and batch normalization).
        model.train(True)

        print(f"Epoch: {epoch + 1}")
        running_loss = 0.0

        # Iterate over the training data loader. The enumerate function provides a count of iterations
        # (batch_index) which is used here for printing the loss periodically.
        for batch_index, batch in enumerate(train_loader):
            # Unpack the current batch of data and move the tensors to the device.
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # Pass the batch through the model to get the output.
            output = model(x_batch)

            # Calculate the loss between the model output and the true labels.
            loss = loss_function(output, y_batch)

            # Add the current loss to the running loss total.
            running_loss += loss.item()

            # Reset the gradients in the optimizer before performing backpropagation.
            optimizer.zero_grad()

            # Perform backpropagation to calculate the gradients.
            loss.backward()

            # Update the model's parameters based on the gradients.
            optimizer.step()

            # Every 100 batches, print the average loss and reset the running loss.
            if batch_index % 100 == 99:
                avg_loss_across_batches = running_loss / 100
                print(
                    "Batch {0}, loss: {1:.3f}".format(
                        batch_index + 1, avg_loss_across_batches
                    )
                )
                running_loss = 0.0
        print()

    # In[2688]:

    def validate_one_epoch(model):
        # Set the model to evaluation mode. This is necessary because certain layers like dropout layers
        # and batch normalization layers behave differently during evaluation.
        model.train(False)
        running_loss = 0.0

        # Iterate over the validation (test_loader) data loader.
        # The enumerate function provides a count of iterations (batch_index) which is not used here.
        for batch_index, batch in enumerate(
            test_loader
        ):  # Corrected typo: "enumerate" and should be test_loader
            # Unpack the current batch of data and move the tensors to the device.
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # Disable gradient calculations. This is important as it reduces memory consumption and speeds up computation.
            with torch.no_grad():
                # Pass the batch through the model to get the output.
                output = model(x_batch)

                # Calculate the loss between the model output and the true labels.
                loss = loss_function(output, y_batch)

                # Add the current loss to the running loss total.
                running_loss += loss.item()

        # Calculate the average loss across all batches in the test dataset.
        avg_loss_across_batches = running_loss / len(test_loader)

        # Print the average validation loss.
        print("Val Loss: {0:.3f}".format(avg_loss_across_batches))
        print("***************************************************")
        # Print a newline (likely for formatting purposes).
        print()

    # In[3064]:

    learning_rate = 0.0007
    num_epochs = 100
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(apple_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(apple_model)
        validate_one_epoch(apple_model)

    # In[3065]:

    with torch.no_grad():
        predicted = apple_model(X_train.to(device)).to("cpu").numpy()
        plt.plot(y_train, label="Actual Close")
        plt.plot(predicted, label="Predicted Close")
        plt.xlabel("Day")
        plt.ylabel("Close")
        plt.legend()
        plt.show()

    # In[3066]:

    train_predictions = predicted.flatten()

    dummies = np.zeros((X_train.shape[0], lookback + 1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)

    train_predictions = dc(dummies[:, 0])
    train_predictions

    # In[3067]:

    dummies = np.zeros((X_train.shape[0], lookback + 1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])
    new_y_train

    # In[3068]:

    plt.plot(new_y_train, label="Actual Close")
    plt.plot(train_predictions, label="Predicted Close")
    plt.xlabel("Day")
    plt.ylabel("Close")
    plt.legend()
    plt.show()

    # In[3069]:

    test_predictions = apple_model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    # In[3070]:

    dummies = np.zeros((y_test.shape[0], lookback + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_test = dc(dummies[:, 0])
    new_y_test

    # In[3071]:

    plt.plot(new_y_test, label="Actual Close")
    plt.plot(test_predictions, label="Predicted Close")
    plt.xlabel("Day")
    plt.ylabel("Close")
    plt.legend()
    plt.show()

    # In[3072]:

    def predict_next_7_days_close(model, last_seq, scaler, device):
        model.eval()  # Set the model to evaluation mode

        future_prices = []
        # Reshape last_seq to 2D if it's not already
        current_seq = (
            np.array(last_seq).reshape(1, -1)
            if last_seq.ndim == 1
            else np.array(last_seq)
        )

        for _ in range(7):
            # Scale the last sequence
            current_seq_scaled = scaler.transform(current_seq)

            # Convert to tensor and reshape to match the input shape of the model which is 3d with .view(1, -1, 1), -1 is the placeholder for
            # current_seq_scaled length which is 7 t-1 --> t-7
            current_seq_tensor = (
                torch.tensor(current_seq_scaled[:, -7:], dtype=torch.float32)
                .view(1, -1, 1)
                .to(device)
            )

            # Predict the next day close price
            with torch.no_grad():
                predicted_close_scaled = model(current_seq_tensor).cpu().numpy()

            # Prepare a dummy array for inverse scaling
            inverse_scaling_input = np.zeros((1, scaler.n_features_in_))
            inverse_scaling_input[0, -1] = predicted_close_scaled[
                0, 0
            ]  # Set the last value to the predicted price

            # Inverse scale the predicted close price
            predicted_close = scaler.inverse_transform(inverse_scaling_input)[0, -1]

            # Append the predicted close price to the list of future prices
            future_prices.append(predicted_close)

            # The current_seq array is being shifted to the left to make room for the new predicted closing price, effectively
            # creating a new sequence for the next prediction. This process is like sliding a window over a timeline of data,
            # where the window moves forward by one step (one day in this case) each time a new prediction is made.
            # Update the current sequence to include the predicted price
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[
                0, -1
            ] = predicted_close  # Replace the oldest price with the newest predicted price

        return future_prices[::-1]

    def predict_and_save():
        # Assuming new_y_train, train_predictions, new_y_test, test_predictions are defined

        # Predict next 7 days close prices
        last_7_days_closing_prices = shifted_df_apple.tail(1)
        predictions = predict_next_7_days_close(
            apple_model, last_7_days_closing_prices, scaler, device
        )

        # Save predictions to a JSON file
        json_output = {"predictions": predictions}
        with open("../static/predictionsAMZN.json", "w") as json_file:
            json.dump(json_output, json_file)

        # Save predictions to a TXT file
        with open("predictions.txt", "w") as txt_file:
            for day, prediction in enumerate(predictions, start=1):
                txt_file.write(f"Day {day}: {prediction}\n")

        # Return a response with the predictions
        return jsonify({"predictions": predictions})

    # In[3073]:

    apple_last_7_days = shifted_df_apple.tail(1)
    apple_last_7_days_np = apple_last_7_days.to_numpy()
    last_7_days_closing_prices = apple_last_7_days
    predictions = predict_next_7_days_close(
        apple_model, last_7_days_closing_prices, scaler, device
    )
    print(predictions)

    # In[3074]:

    def print_next_7_days_close_graph(actual_prices, predictions):
        plt.figure(figsize=(10, 5))

        # Plot the last 1 month actual close prices
        plt.plot(actual_prices, label="Actual Close", color="blue")

        # Add the last actual price to the beginning of the predictions to connect the lines
        connected_predictions = [actual_prices[-1]] + predictions

        # Generate the range for the connected predictions
        connected_days = range(
            len(actual_prices) - 1, len(actual_prices) - 1 + len(connected_predictions)
        )

        # Plot the connected predictions
        plt.plot(
            connected_days,
            connected_predictions,
            label="Predicted Close",
            color="orange",
        )

        plt.xlabel("Day")
        plt.ylabel("Close Price")
        plt.title("Actual and Predicted Close Prices")
        plt.legend()
        plt.show()

    # In[3075]:

    def prepare_apple_dataframe():
        apple = yf.Ticker("AAPL")
        apple = apple.history(period="max")
        apple = apple.loc["2000-01-01":].copy()

        # Assuming you have a DataFrame 'df' with the close prices and a date index
        last_month_prices = apple["Close"][
            -30:
        ].tolist()  # Get the last 30 days of actual close prices

        apple_last_7_days = shifted_df_apple.tail(1)
        apple_last_7_days_np = apple_last_7_days.to_numpy()
        last_7_days_closing_prices = apple_last_7_days

        predictions = predict_next_7_days_close(
            apple_model, last_7_days_closing_prices, scaler, device
        )
        print_next_7_days_close_graph(last_month_prices, predictions)

    # In[3076]:

    prepare_apple_dataframe()

    # In[3077]:

    # Assuming new_y_train, train_predictions, new_y_test, test_predictions are defined

    # Calculate MSE and MAE for the training data
    mse_train = mean_squared_error(new_y_train, train_predictions)
    mae_train = mean_absolute_error(new_y_train, train_predictions)

    # Calculate MSE and MAE for the test data
    mse_test = mean_squared_error(new_y_test, test_predictions)
    mae_test = mean_absolute_error(new_y_test, test_predictions)

    print("Training Data - MSE:", mse_train, "MAE:", mae_train)
    print("Test Data - MSE:", mse_test, "MAE:", mae_test)

    # In[3078]:

    # Save the model state dictionary
    torch.save(apple_model.state_dict(), "apple_model.pth")

    # In[ ]:

    # In[ ]:

    # For demonstration purposes, a simple output
    output_text = "Hello, this is your Flask web app!"

    # Generate predictions
    predictions = predict_next_7_days_close(
        apple_model, last_7_days_closing_prices, scaler, device
    )

    # Create a plot
    plt.figure(figsize=(10, 5))
    plt.plot(new_y_test, label="Actual Close", color="blue")
    plt.plot(
        range(len(new_y_test), len(new_y_test) + len(predictions)),
        predictions,
        label="Predicted Close",
        color="orange",
    )
    plt.xlabel("Day")
    plt.ylabel("Close Price")
    plt.title("Actual and Predicted Close Prices")
    plt.legend()

    # Save the plot as an image
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    predictions_response = predict_and_save()

    try:
        # Check if the response is successful before using json()
        predictions_response.raise_for_status()
        predictions = predictions_response.json()["predictions"]

        # You can use the 'predictions' data as needed in your index route
        # For example, pass it to the template or use it in your visualization logic

        # For demonstration purposes, a simple output
        output_text = "Hello, this is your Flask web app!"

        # Generate predictions
        predictions = predict_next_7_days_close(
            apple_model, last_7_days_closing_prices, scaler, device
        )

        # Create a plot
        plt.figure(figsize=(10, 5))
        plt.plot(new_y_test, label="Actual Close", color="blue")
        plt.plot(
            range(len(new_y_test), len(new_y_test) + len(predictions)),
            predictions,
            label="Predicted Close",
            color="orange",
        )
        plt.xlabel("Day")
        plt.ylabel("Close Price")
        plt.title("Actual and Predicted Close Prices")
        plt.legend()

        # Save the plot as an image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Render the HTML template with the actual and predicted prices and the embedded image
        return render_template(
            "dog.html",
            actual_and_predicted_prices=list(
                zip(range(1, len(predictions) + 1), new_y_test, predictions)
            ),
            plot_url=plot_url,  # Pass the base64-encoded image to the template
        )
    except Exception as e:
        # Handle the case when there is an error accessing the 'predictions' key
        print(f"Error accessing 'predictions' key: {e}")
        return "Error accessing 'predictions' key"



if __name__ == "__main__":
    app.run(debug=True)
