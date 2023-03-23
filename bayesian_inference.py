import numpy as np
from maxflow.fastmin import aexpansion_grid
from maxflow import Graph
# Load the image of probabilities
img_prob = np.load('/nfs/a1/homes/py15jmc/bootstrap/2023/timeseries/probabilities_100_101_20230221_202044.npy')

# Define the parameters for the Markov random field
lambda_ = 1.0  # regularization parameter
gamma = 50.0  # parameter for the likelihood function
num_labels = 2  # number of labels (changed or unchanged)

wdir = "/nfs/a1/homes/py15jmc/"
# Define the neighbor coordinates for each pixel
neighbor_i_coords = np.load(wdir + "bootstrap/shp_i_20150511_20170617.npy")
neighbor_j_coords = np.load(wdir + "bootstrap/shp_j_20150511_20170617.npy")


def segment_image(image, beta, num_iterations, neighbor_i_coords=None, neighbor_j_coords=None):
    height, width = image.shape

    # Create a graph with height x width nodes
    g = Graph[float](height * width, height * width * 4)

    # Add nodes to the graph
    nodes = g.add_nodes(height * width)

    # Add edges to the graph
    for i in range(height):
        for j in range(width):
            node_id = i * width + j
            pixel_prob = image[i, j]

            # Add terminal edges to the source and sink nodes
            g.add_tedge(node_id, -np.log(pixel_prob), -np.log(1 - pixel_prob))

            # Define the neighbors of each pixel
            if neighbor_i_coords is not None and neighbor_j_coords is not None:
                mask = ~np.isnan(neighbor_i_coords[:, i, j]) * ~np.isnan(neighbor_j_coords[:, i, j])

                neighbors = zip(neighbor_i_coords[:, i, j][mask], neighbor_j_coords[:, i, j][mask])


            else:
                neighbors = [(i-1, j), (i, j-1), (i, j+1), (i+1, j)]

            # Check each neighbor and add an edge to the graph if the neighbor is within the image boundaries
            for neighbor in neighbors:
                if neighbor[0] >= 0 and neighbor[0] < height and neighbor[1] >= 0 and neighbor[1] < width:
                    neighbor_id = neighbor[0] * width + neighbor[1]
                    neighbor_prob = image[int(neighbor[0]), int(neighbor[1])]

                    # Add an edge between the current node and the neighboring node
                    edge_weight = beta * np.exp(-(pixel_prob - neighbor_prob) ** 2 / (2 * (0.1 ** 2)))
                    g.add_edge(node_id, neighbor_id, edge_weight, edge_weight)

    # Run alpha-expansion to optimize the graph
    labels = aexpansion_grid(g, beta)

    # Reshape the labels into an image
    segmented_image = labels.reshape(height, width)

    return segmented_image


seg = segment_image(img_prob, 1, 5, neighbor_i_coords, neighbor_j_coords)

plt.pcolormesh(seb)
plt.colorbar()
plt.savefig("markovrandomfield_image.jpg")
# 1 = foreground -> changed