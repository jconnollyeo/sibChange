import numpy as np
from maxflow import Graph

# Load the image of probabilities
img_prob = np.load('/nfs/a1/homes/py15jmc/bootstrap/2023/timeseries/predictions_100_101_20230221_202044.npy')

# Define the parameters for the Markov random field
lambda_ = 1.0  # regularization parameter
gamma = 50.0  # parameter for the likelihood function
num_labels = 2  # number of labels (changed or unchanged)

wdir = "/nfs/a1/homes/py15jmc/"
# Define the neighbor coordinates for each pixel
neighbor_i_coords = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
neighbor_j_coords = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")

# Define the graph
height, width = img_prob.shape
num_nodes = height * width
graph = Graph[int](num_nodes, num_nodes * len(neighbor_i_coords[0]))

# Add the nodes and their unary potentials
for y in range(height):
    for x in range(width):
        node_id = y * width + x
        prob = img_prob[y, x]
        graph.add_nodes(node_id, prob, 1.0 - prob)

# Add the edges and their pairwise potentials
for y in range(height):
    for x in range(width):
        node_id = y * width + x
        for k in range(len(neighbor_i_coords[0])):
            neighbor_y = y + neighbor_i_coords[node_id, k]
            neighbor_x = x + neighbor_j_coords[node_id, k]
            if neighbor_y >= 0 and neighbor_y < height and neighbor_x >= 0 and neighbor_x < width:
                neighbor_id = neighbor_y * width + neighbor_x
                diff = img_prob[y, x] - img_prob[neighbor_y, neighbor_x]
                weight = gamma * np.exp(-lambda_ * diff**2)
                graph.add_edge(node_id, neighbor_id, weight, weight)

# Find the minimum energy labeling using alpha expansion
graph.maxflow()
labeling = graph.get_labeling()

# Convert the labeling to a binary image of changed or unchanged pixels
img_bin = labeling.reshape(height, width)


plt.pcolormesh(img_bin)
plt.colorbar()

# Save the binary image to a file
np.save('binary_image.npy', img_bin)
