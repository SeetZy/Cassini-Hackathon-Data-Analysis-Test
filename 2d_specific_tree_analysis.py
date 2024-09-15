import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial.distance import cdist

rasterfile = r"raster.tif"
ds = rasterio.open(rasterfile, "r")

# Parameters
min_z_threshold = 35  # Minimum Z height for trees (above sea level)
# Maximum possible tree height (pines typically grow to 40-45 meters)
max_z_threshold = 36
fixed_rectangle_size = 20  # Initial fixed size for tree rectangles (in pixels)
# Max distance between trees to be considered part of the same cluster
distance_threshold = 25

# Read full raster for visualization and analysis
# Use masked=True to automatically handle NaN values
full_raster = ds.read(1, masked=True)
height, width = ds.height, ds.width
transform = ds.transform

# Normalize raster to grayscale for OpenCV processing (255 scale)
raster_image = (full_raster.data * (255 / full_raster.max())).astype(np.uint8)
# Use binary thresholding to isolate potential trees based on Z height
_, binary_thresh = cv2.threshold(raster_image, 50, 255, cv2.THRESH_BINARY)

# Apply morphological operations to remove noise and small objects
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(binary_thresh, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)
# Detect contours for small tree-like structures
contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
tree_data = []

# Go through each contour, which represents an individual tree
for contour in contours:
    area = cv2.contourArea(contour)

    # Filter small areas (likely noise, not trees)
    if area < 50:  # Adjust this threshold to filter smaller non-tree contours
        continue

    # Get the center of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:  # Skip if there's a divide-by-zero issue
        continue
    cx = int(M["m10"] / M["m00"])  # X coordinate of the contour's center
    cy = int(M["m01"] / M["m00"])  # Y coordinate of the contour's center

    # Determine the coordinates of the fixed-size rectangle around the tree
    x = max(0, cx - fixed_rectangle_size // 2)
    y = max(0, cy - fixed_rectangle_size // 2)
    w = fixed_rectangle_size
    h = fixed_rectangle_size
    # Extract the Z values for the region of the tree contour from the original raster
    tree_region = full_raster[y:y+h, x:x+w]

    # Ignore regions that are masked (no data)
    if np.ma.is_masked(tree_region) or np.isnan(tree_region).all():
        continue

    # Calculate the height of the tree: Z value (tree top) - Z value (terrain)
    max_z_value = np.nanmax(tree_region)  # Highest Z value in the tree region
    # Minimum Z value in the tree region (terrain level)
    min_z_value = np.nanmin(tree_region)
    # Apply Z height thresholding to exclude outliers
    tree_height = max_z_value - min_z_value

    if tree_height < min_z_threshold or tree_height > max_z_threshold:
        continue

    # Store the tree data (coordinates and height)
    tree_data.append({
        "x": x + fixed_rectangle_size // 2,  # Store the center coordinates
        "y": y + fixed_rectangle_size // 2,
        "tree_height_meters": tree_height
    })

# --- Cluster the trees based on proximity ---

# Extract the coordinates of each detected tree
tree_coordinates = np.array([[tree["x"], tree["y"]] for tree in tree_data])

# Use a distance matrix to group trees that are close to each other
if len(tree_coordinates) > 1:
    dist_matrix = cdist(tree_coordinates, tree_coordinates)
    clusters = []
    # Track clustered trees
    clustered = np.zeros(len(tree_coordinates), dtype=bool)

    for i, coord in enumerate(tree_coordinates):
        if clustered[i]:
            continue
        # Find all trees within the distance threshold
        cluster = np.where(dist_matrix[i] < distance_threshold)[0]
        clustered[cluster] = True
        clusters.append(cluster)
else:
    clusters = [[0]]

# --- Visualization of Results ---

fig, ax = plt.subplots(figsize=(10, 10))
image = ax.imshow(raster_image, cmap='cividis', extent=(
    # Plot the full raster as an image in grayscale
    0, width, 0, height), aspect='equal')

# Loop over the clusters and display the average tree height and tree count beside the cluster
for cluster in clusters:
    # Get the average height for the current cluster
    cluster_heights = [tree_data[i]["tree_height_meters"] for i in cluster]

    # Apply post-clustering height thresholding: Ensure all tree heights in the cluster are within the valid range
    if np.any(np.array(cluster_heights) < min_z_threshold) or np.any(np.array(cluster_heights) > max_z_threshold):
        continue

    average_height = np.mean(cluster_heights)
    num_trees = len(cluster)  # Number of trees in the cluster

    # Get the centroid of the cluster to place the text
    cluster_coords = tree_coordinates[cluster]
    cluster_centroid = np.mean(cluster_coords, axis=0)

    # Determine the rectangle size based on the spread of the cluster
    min_x, min_y = np.min(cluster_coords, axis=0)
    max_x, max_y = np.max(cluster_coords, axis=0)
    cluster_width = max_x - min_x + fixed_rectangle_size
    cluster_height = max_y - min_y + fixed_rectangle_size

    # Draw a rectangle around the cluster
    rect = Rectangle(
        (min_x - fixed_rectangle_size // 2, min_y - fixed_rectangle_size // 2),
        cluster_width, cluster_height,
        linewidth=1.0, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

    # Annotate with "Avg" and the height for clusters with more than 1 tree, or just the height if 1 tree
    if num_trees == 1:
        label = f'{average_height:.1f} m'
    else:
        label = f'Avg\n{average_height:.1f} m ({num_trees})'

    ax.text(cluster_centroid[0] + cluster_width // 2 + 5, cluster_centroid[1],
            label, color='yellow', fontsize=6, verticalalignment='center')

# Add grid lines based on the real-world 100x100 meter grid
x_min, x_max = transform[2], transform[2] + width * transform[0]
y_min, y_max = transform[5], transform[5] + height * transform[4]

x_ticks_real_world = np.arange(np.floor(x_min), np.ceil(x_max), 100)
y_ticks_real_world = np.arange(np.floor(y_max), np.ceil(y_min), 100)

x_ticks_pixel = (x_ticks_real_world - transform[2]) / transform[0]
y_ticks_pixel = height - (y_ticks_real_world - transform[5]) / transform[4]

ax.set_xticks(x_ticks_pixel, labels=[f"{int(x)}" for x in x_ticks_real_world])
ax.set_yticks(y_ticks_pixel, labels=[f"{int(y)}" for y in y_ticks_real_world])

ax.set_xlabel('X Coordinates (meters)')
ax.set_ylabel('Y Coordinates (meters)')
ax.set_aspect('equal')

plt.colorbar(image, ax=ax, label='Z values in meters (elevation)')
plt.title(f'Raster Tree Height Analysis: {rasterfile}')

plt.show()

# Output the tree data with their corresponding heights and tree counts
for i, cluster in enumerate(clusters, start=1):
    cluster_heights = [tree_data[j]["tree_height_meters"] for j in cluster]
    if np.any(np.array(cluster_heights) < min_z_threshold) or np.any(np.array(cluster_heights) > max_z_threshold):
        continue
    average_height = np.mean(cluster_heights)
    num_trees = len(cluster)
