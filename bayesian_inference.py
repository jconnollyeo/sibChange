import numpy as np
from maxflow.fastmin import aexpansion_grid

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

def markov_random_field(image, num_neighbors):
    h, w = image.shape
    num_nodes = h * w
    num_edges = 2 * num_nodes # each node can have up to 4 neighbors
    
    g = Graph[int](num_nodes, num_edges)
    
    for node_index in range(num_nodes):
        g.add_node(node_index)
    
    for i in range(h):
        for j in range(w):
            node_index = i * w + j
            
            # add edges to neighbors
            for di, dj in num_neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbor_index = ni * w + nj
                    diff = abs(image[i, j] - image[ni, nj])
                    capacity = 1.0 / (1.0 + diff)
                    g.add_edge(node_index, neighbor_index, capacity, capacity)
    
    # compute maximum flow
    flow = g.maxflow()
    
    # return segmented image
    segmented_image = np.zeros_like(image)
    for i in range(h):
        for j in range(w):
            node_index = i * w + j
            if g.get_segment(node_index) == Graph.SINK:
                segmented_image[i, j] = 1
    
    return segmented_image

seg = markov_random_field(img_prob, 2)
# 1 = foreground -> changed