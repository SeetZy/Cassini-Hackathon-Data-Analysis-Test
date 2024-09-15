import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rasterfile = r"raster.tif"

try:
    with rasterio.open(rasterfile, "r") as ds:
        raster_data = ds.read(1)

        rows, cols = np.meshgrid(np.arange(ds.height),
                                 np.arange(ds.width), indexing='ij')

        xs, ys = rasterio.transform.xy(ds.transform, rows, cols)

        xcoords = np.array(xs)
        ycoords = np.array(ys)
        zcoords = raster_data

        zcoords = np.where(zcoords == ds.nodata, np.nan, zcoords)

        x_flat = xcoords.flatten()
        y_flat = ycoords.flatten()
        z_flat = zcoords.flatten()

        valid = ~np.isnan(z_flat)
        x_flat = x_flat[valid]
        y_flat = y_flat[valid]
        z_flat = z_flat[valid]

        flag_mask = z_flat > 40

        fig = plt.figure(figsize=(12, 10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')

        scatter = ax.scatter(x_flat[~flag_mask], y_flat[~flag_mask], z_flat[~flag_mask],
                             c=z_flat[~flag_mask], cmap='RdYlGn_r', marker='.', s=1)

        ax.scatter(x_flat[flag_mask], y_flat[flag_mask], z_flat[flag_mask],
                   c='red', marker='.', s=1)

        ax.set_xlabel("X Coordinate", color='white')
        ax.set_ylabel("Y Coordinate", color='white')
        ax.set_zlabel("Z Height", color='white')

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        ax.view_init(elev=30, azim=120)

        x_range = [np.min(x_flat), np.max(x_flat)]
        y_range = [np.min(y_flat), np.max(y_flat)]
        z_range = [np.min(z_flat), np.max(z_flat)]

        ax.set_xlim([np.mean(x_range) - 0.5 * (x_range[1] - x_range[0]),
                    np.mean(x_range) + 0.5 * (x_range[1] - x_range[0])])
        ax.set_ylim([np.mean(y_range) - 0.5 * (y_range[1] - y_range[0]),
                    np.mean(y_range) + 0.5 * (y_range[1] - y_range[0])])
        ax.set_zlim([np.mean(z_range) - 0.5 * (z_range[1] - z_range[0]),
                    np.mean(z_range) + 0.5 * (z_range[1] - z_range[0])])

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('Elevation (Z Value)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
