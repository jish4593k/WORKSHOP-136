import numpy as np
from sklearn.cluster import KMeans
import cv2
from jinja2 import Template
from scipy.cluster.vq import kmeans, vq
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def get_hex_at_pos(img, x, y):
    color = img[x, y]
    return "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])


def get_matrix(filename):
    img = cv2.imread(filename)
    return img.reshape((-1, 3))


def comp_avg(cluster, matrix):
    pixels = matrix[cluster]
    return "#{:02x}{:02x}{:02x}".format(*np.mean(pixels, axis=0).astype(int))

# Function to convert RGB to hexadecimal
def rbg_to_hex(rgb_list):
    return "#{:02x}{:02x}{:02x}".format(*rgb_list)

CLSTERS = 9


matrix = get_matrix("ryuuko.jpg")

kmeans = KMeans(n_clusters=CLUSTERS, random_state=0).fit(matrix)



# Compute average colors for each cluster
average_colors = [comp_avg(np.where(kmeans.labels_ == i)[0], matrix) for i in range(CLUSTERS)]

# HTML template for visualization
html_template = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { display: flex; }
        .color-box { width: 100px; height: 100px; margin: 5px; border: 1px solid black; }
    </style>
</head>
<body>
    {% for color in color_list %}
        <div class="color-box" style="background-color: {{ color }};"></div>
    {% endfor %}
</body>
</html>
"""

# Generate HTML using Jinja2 template
template = Template(html_template)
html_content = template.render(color_list=average_colors)

# Save HTML content to a file
with open("palette.html", "w") as f:
    f.write(html_content)

# PyTorch example for deep learning
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, CLUSTERS)
        )
        self.decoder = nn.Sequential(
            nn.Linear(CLUSTERS, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


tensor_matrix = torch.tensor(matrix, dtype=torch.float32)


scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(matrix)


autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

num_epochs = 50
for epoch in range(num_epochs):
    outputs = autoencoder(tensor_matrix)
    loss = criterion(outputs, tensor_matrix)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

encoded_output = autoencoder.encoder(tensor_matrix).detach().numpy()
kmeans_centroids, kmeans_labels = kmeans(encoded_output, CLUSTERS)
autoencoder_labels = vq(encoded_output, kmeans_centroids)[0]
