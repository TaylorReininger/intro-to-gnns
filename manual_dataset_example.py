import torch
from torch_geometric.data import Data

"""
In this script, we are simply leveraging the torch_geometric graph data formatting standard. 
This example comes straight from the pytorch-geometric documentation, but includes comments 
to help better understand the content. 

Source: 
https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

We are going to make our own dataset for a 3-node, 4-edge graph, by specifying the nodes 
and edges manually. 
"""

# The nodes are designated by 1 row per node, each containing a list of features
# for the node (in this case 1 feature per node).
# data.x: Node feature matrix with shape [num_nodes, num_node_features]
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
# Node 0's feature is "-1"
# Node 1's feature is "0"
# Node 2's feature is "1"

# The edges are designated by 2 rows, 1 for src nodes and 1 for dst nodes
# The edges are from the top row to the bottom row, and must be listed both ways if the edge is bi-directional
# data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
# Edge 0: Node 0 -> Node 1
# Edge 1: Node 1 -> Node 0
# Edge 2: Node 1 -> Node 2
# Edge 3: Node 2 -> Node 1

# Add the nodes and edges into the torch_geometric data object
data = Data(x=x, edge_index=edge_index)

# Display the data
print(data)

# Validate the dataset with the built in "validate" method
data.validate(raise_on_error=True)
print('Data successfully validated')
