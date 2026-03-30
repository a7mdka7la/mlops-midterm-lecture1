import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

# 1. Identity: Set a unique experiment name outside the run
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Assignment3_AhmedKahla")

# --- Original Script Setup ---
DATA_PATH = 'data.csv'  # Fixed the hardcoded secret path so it works in Docker

# Read data and fix the deprecated np.float to float
df = pd.read_csv(DATA_PATH)
data = df.values.astype(float) 

# GAN Model Definition
class RandomGAN(nn.Module):
    def __init__(self):
        super(RandomGAN, self).__init__()
        self.gen = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))
        
    def forward(self, x):
        return self.gen(x)

model = RandomGAN()

# Define the hyperparameters you want to test
lr = 0.05
batch_size = len(data)
epochs = 5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# 2. Start the tracking run
with mlflow.start_run():
    
    # 3. Tags: Add an identifier for searching
    mlflow.set_tag("student_id", "AhmedKahla")
    mlflow.set_tag("model_type", "GAN")
    
    # 4. Parameters: Log the "Input" config
    mlflow.log_params({
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "Adam"
    })

    # Prepare data for training
    inputs = torch.tensor(data, dtype=torch.float32)
    labels = torch.zeros_like(inputs) # dummy labels

    # --- YOUR STANDARD PYTORCH TRAINING LOOP GOES HERE ---
    print("Starting training loop...")
    for epoch in range(epochs):
        optimizer.zero_grad() # Clear gradients
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # 5. Live Logging: Inside your loop, log the metrics
        mlflow.log_metric("loss", loss.item(), step=epoch)
    
    print(f'Training complete. Final loss: {loss.item()}')
    
    # 6. Artifact Storage: Log the final model instead of just torch.save()
    mlflow.pytorch.log_model(model, "model")
    print("Model and metrics successfully logged to MLflow!")
