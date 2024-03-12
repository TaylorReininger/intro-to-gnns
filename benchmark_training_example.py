import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

"""
In this script, we are going to benchmark torch_geometric with the Cora dataset
from Planetoid.

Sources: 
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html
https://arxiv.org/abs/1603.08861

We are going to pull down the dataset, train on it, and provide test accuracy. 
"""

# Download the Cora dataset from Planetoid
dataset = Planetoid(root="/tmp/Cora", name="Cora")

# Ensure that the dataset has loaded correctly by checking some features
num_graphs = len(dataset[0].x)
print("Number of examples in the dataset:       %d" % (num_graphs))
print("Number of classes in the dataset:        %d" % (dataset.num_classes))
print("Number of node features in the dataset:  %d" % (dataset.num_node_features))

# Create the Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize two convolutional layers
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        # Extract the node and edge data from the data object
        x = data.x
        edge_index = data.edge_index

        # Pass the data through the first convolutional layer
        x = self.conv1(x, edge_index)
        # Do a ReLu activation
        x = F.relu(x)
        # Do dropout
        x = F.dropout(x, training=self.training)
        # Pass the data through the second convolutional layer
        x = self.conv2(x, edge_index)
        # Take the softmax of the output
        y_pred = F.log_softmax(x, dim=1)
        return y_pred

# Determine which device will be used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Put the network on the device memory
model = GCN().to(device)
# Put the dataset on the device memory
data = dataset[0].to(device)

# Create the optimizer for training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Tell the model you are going to be training
model.train()

# Train the model and save the loss
training_loss = []
for epoch in range(400):
    # Reset the gradients of any tensors (just to be safe)
    optimizer.zero_grad()
    # Perform a forward pass
    out = model(data)
    # Calculate the loss
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # Store the loss in the training loss record
    training_loss.append(loss.item())
    # Perform back propogation
    loss.backward()
    # Step the optimizer forward
    optimizer.step()

# Display the training loss
plt.plot(training_loss)
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Tell the model you are going to be evaluating (don't perform dropout or batch norms)
model.eval()
# Perform inference on the entire dataset
pred = model(data).argmax(dim=1)
# Get the accuracy on the test data
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
# Get the loss on the test data
test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
ave_test_loss = loss.item()

# Display the test accuracy and loss
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Loss: {ave_test_loss:.6f}")
