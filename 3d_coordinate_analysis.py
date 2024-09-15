import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path to the raster file
rasterfile = r"raster.tif"

try:
    # Open the raster file for reading
    with rasterio.open(rasterfile, "r") as ds:
        # Read the data from the first band of the raster
        raster_data = ds.read(1)

        # Create meshgrid for rows and columns
        rows, cols = np.meshgrid(np.arange(ds.height),
                                 np.arange(ds.width), indexing='ij')

        # Transform row and column indices to geographic coordinates
        xs, ys = rasterio.transform.xy(ds.transform, rows, cols)

        # Convert lists of x and y coordinates to numpy arrays
        xcoords = np.array(xs)
        ycoords = np.array(ys)
        # Z coordinates are directly from raster data
        zcoords = raster_data

        # Replace nodata values with NaN for better visualization
        zcoords = np.where(zcoords == ds.nodata, np.nan, zcoords)

        # Flatten the coordinate arrays
        x_flat = xcoords.flatten()
        y_flat = ycoords.flatten()
        z_flat = zcoords.flatten()

        # Filter out NaN values
        valid = ~np.isnan(z_flat)
        x_flat = x_flat[valid]
        y_flat = y_flat[valid]
        z_flat = z_flat[valid]

        # Flag for points where z value is greater than 40
        flag_mask = z_flat > 40

        # Set up the figure and 3D axis for plotting
        fig = plt.figure(figsize=(12, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')

        # Scatter plot for points with z values <= 40
        scatter = ax.scatter(x_flat[~flag_mask], y_flat[~flag_mask], z_flat[~flag_mask],
                             c=z_flat[~flag_mask], cmap='RdYlGn_r', marker='.', s=1)

        # Scatter plot for points with z values > 40 in red
        ax.scatter(x_flat[flag_mask], y_flat[flag_mask], z_flat[flag_mask],
                   c='red', marker='.', s=1)

        # Set axis labels and colors
        ax.set_xlabel("X Coordinate", color='white')
        ax.set_ylabel("Y Coordinate", color='white')
        ax.set_zlabel("Z Height", color='white')

        # Set tick parameters to match the color of the labels
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        # Set the viewing angle of the plot
        ax.view_init(elev=30, azim=120)

        # Define the range for each axis
        x_range = [np.min(x_flat), np.max(x_flat)]
        y_range = [np.min(y_flat), np.max(y_flat)]
        z_range = [np.min(z_flat), np.max(z_flat)]

        # Adjust the limits of each axis for better visualization
        ax.set_xlim([np.mean(x_range) - 0.5 * (x_range[1] - x_range[0]),
                    np.mean(x_range) + 0.5 * (x_range[1] - x_range[0])])
        ax.set_ylim([np.mean(y_range) - 0.5 * (y_range[1] - y_range[0]),
                    np.mean(y_range) + 0.5 * (y_range[1] - y_range[0])])
        ax.set_zlim([np.mean(z_range) - 0.5 * (z_range[1] - z_range[0]),
                    np.mean(z_range) + 0.5 * (z_range[1] - z_range[0])])

        # Add a colorbar to the plot
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Elevation (Z Value)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        # Display the plot
        plt.show()

except Exception as e:
    # Print error message if an exception occurs
    print(f"An error occurred: {e}")
