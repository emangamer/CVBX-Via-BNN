import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BoundingBoxNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BoundingBoxNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

def generate_dataset(num_samples, num_vertices, holdout_ratio=0.2):
    # Generate random Gaussian splat data
    gaussian_data = np.random.normal(0, 1, (num_samples, num_vertices))

    # Generate random bounding box parameters
    bounding_boxes = np.random.rand(num_samples, 6)  # [x, y, z, width, height, depth]

    # Split the dataset into training and testing sets
    train_size = int(num_samples * (1 - holdout_ratio))
    train_gaussian_data = gaussian_data[:train_size]
    train_bounding_boxes = bounding_boxes[:train_size]
    test_gaussian_data = gaussian_data[train_size:]
    test_bounding_boxes = bounding_boxes[train_size:]

    return train_gaussian_data, train_bounding_boxes, test_gaussian_data, test_bounding_boxes

def train_network(model, train_gaussian_data, train_bounding_boxes, epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for gaussian_data, bounding_box in zip(train_gaussian_data, train_bounding_boxes):
            gaussian_data_tensor = torch.tensor(gaussian_data, dtype=torch.float32)
            bounding_box_tensor = torch.tensor(bounding_box, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(gaussian_data_tensor)
            loss = criterion(output, bounding_box_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/len(train_gaussian_data):.4f}')

    print(f'Final Loss: {total_loss/len(train_gaussian_data):.6f}')

def test_network(model, test_gaussian_data, test_bounding_boxes):
    model.eval()
    with torch.no_grad():
        for gaussian_data, bounding_box in zip(test_gaussian_data, test_bounding_boxes):
            gaussian_data_tensor = torch.tensor(gaussian_data, dtype=torch.float32)

            output = model(gaussian_data_tensor)
            predicted_bounding_box = output.numpy()

            print("Gaussian Splat Data:")
            print(gaussian_data)
            print("Predicted Bounding Box:")
            print(predicted_bounding_box)
            print("Actual Bounding Box:")
            print(bounding_box)
            print("---")

# Set the parameters for the dataset and model
num_samples = 1000
num_vertices = 100
input_size = num_vertices
hidden_size = 256
output_size = 6  # Bounding box parameters: [x, y, z, width, height, depth]

# Generate the dataset
train_gaussian_data, train_bounding_boxes, test_gaussian_data, test_bounding_boxes = generate_dataset(num_samples, num_vertices)

# Create and train the model
model = BoundingBoxNN(input_size, hidden_size, output_size)
train_network(model, train_gaussian_data, train_bounding_boxes)

# Test the model
test_network(model, test_gaussian_data, test_bounding_boxes)
