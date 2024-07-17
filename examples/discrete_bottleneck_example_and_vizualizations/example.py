import numpy as np
import torch
from vqvae import VQVAEDiscreteLayer 

embedding_dim = 2
X = torch.randn(100, 1, embedding_dim, requires_grad=True) # batch_size, seq_len, embedding_dim
print("X:", X) 
print("X.shape:", X.shape) # (10, 5, 32)

# Create a discrete layer
params = {
    'temperature': 1.0,
    'label_smoothing_scale': 0.0,
    "dist_ord": 2,
    'vocab_size': 6,
    'dictionary_dim': embedding_dim,
    'hard': True,
    'projection_method': "None", # "unit-sphere" "scale" "layer norm" or "None"
    'beta': 5
}

discrete_layer = VQVAEDiscreteLayer(**params)
print("discrete_layer:", discrete_layer)

# Discretize the input
indices, probs, quantized, vq_loss = discrete_layer.discretize(X)
print("indices:", indices)
print("probs:", probs)
print("quantized:", quantized)
print("dic:", discrete_layer.dictionary.weight)
print("vq_loss:", vq_loss)

### --- Visualization --- ###

# Assumes X and dictionary are 2D points, and uses VQVAEDiscreteLayer to 
# optimize their positions and minimize distance to cluster representative 

import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import torch.nn as nn
import torch.optim as optim
import imageio
import os
import torch.nn.functional as F
from scipy.spatial import Voronoi, voronoi_plot_2d

dic = discrete_layer.dictionary.weight

shutil.rmtree('figs/', ignore_errors=True)
os.makedirs('figs/')

def plot_assigments(X, dic, indices, epoch=0):
    X = X[:,0,:].numpy()
    dic = dic.detach().numpy()
    indices = indices[:,0].numpy()

    # Plot Voronoi of dictionary vertices 
    vor = Voronoi(dic)
    voronoi_plot_2d(vor, show_vertices=False, show_points=False)

    # Plot points in X
    plt.scatter(X[:, 0], X[:, 1], c=indices, cmap='viridis', label='Data Points')

    # Plot points in dic
    plt.scatter(dic[:, 0], dic[:, 1], c='red', marker='X', s=100, label='Center Points')

    # Connect each point in X to its corresponding center point in dic
    for i, (x, y) in enumerate(X):
        center_x, center_y = dic[indices[i]]
        plt.plot([x, center_x], [y, center_y], color='gray', linestyle='--', linewidth=0.5)

    plt.title(f'Visualization of 2D Points and Center Points (Epoch {epoch})')
    plt.xlabel('X-axis') 
    plt.ylabel('Y-axis')
    plt.legend()
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('equal')
    plt.savefig(f'figs/epoch_{str(epoch + 1).zfill(6)}_visualization.png')
    plt.close()

criterion = nn.MSELoss()

optimizer = optim.SGD([
    {'params': discrete_layer.parameters()},
    {'params': [X], 'lr': 0.5}
], lr=0.5)

def repulsion_loss(points, min_distance=0.1):
    num_points = points.size(0)
    distances = torch.cdist(points, points)
    mask = ~torch.eye(num_points, dtype=torch.bool, device=points.device)
    distances = distances[mask]
    loss = F.relu(min_distance - distances).sum()
    return loss

num_epochs = 1000
for epoch in range(num_epochs):
    discrete_layer.train()  
    indices, probs, quantized, vq_loss = discrete_layer.discretize(X)

    dic = discrete_layer.dictionary.weight
    dic_loss = repulsion_loss(dic, min_distance=0.0)
    loss = vq_loss + dic_loss * 100000

    optimizer.zero_grad(discrete_layer.parameters)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    with torch.no_grad():
        if epoch % 10 == 0:
            plot_assigments(X, dic, indices, epoch=epoch)

# Save plots to video
image_dir = "figs/"
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_visualization.png')])
video_writer = imageio.get_writer('visualization_video.mp4', fps=25)
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = imageio.v2.imread(image_path)
    video_writer.append_data(image)
video_writer.close()
