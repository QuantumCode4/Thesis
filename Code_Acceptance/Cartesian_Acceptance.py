import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_spatial_acceptance(x0, y0, theta, phi, accepted_mask, L, D, N_planes, plane_index, num_bins):
    if not (0 <= plane_index < N_planes):
        raise ValueError(f"The index plane is between 0 and {N_planes - 1}")

    tan_theta = np.tan(theta)
    dx = plane_index * D * tan_theta * np.cos(phi)
    dy = plane_index * D * tan_theta * np.sin(phi)
    
    x_plane = x0 + dx
    y_plane = y0 + dy

    x_plane_acc = x_plane[accepted_mask]
    y_plane_acc = y_plane[accepted_mask]

    edges = np.linspace(-L/2, L/2, num_bins)

    H_gen_xy, _, _ = np.histogram2d(x_plane, y_plane, bins=(edges, edges))
    H_acc_xy, _, _ = np.histogram2d(x_plane_acc, y_plane_acc, bins=(edges, edges))

    acceptance_xy = np.divide(
        H_acc_xy,
        H_gen_xy,
        out=np.zeros_like(H_acc_xy, dtype=float),
        where=H_gen_xy != 0
    )

    return acceptance_xy, edges

def plot_acceptance(x0, y0, theta, phi, accepted_mask,
                    L, D, N_planes, plane_index, NUM_BINS):

    acceptance_xy, edges = compute_spatial_acceptance(
        x0, y0, theta, phi, accepted_mask,
        L, D, N_planes, plane_index, NUM_BINS
    )

    fig = plt.figure(figsize=(14, 6))

    ax2d = fig.add_subplot(1, 2, 1)
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    im = ax2d.imshow(
        acceptance_xy.T,
        extent=extent,
        cmap='plasma',
        interpolation='nearest',
        origin='lower',
        vmin=0, vmax=1
    )
    ax2d.set_title(f"2D Spatial Acceptance Map - Plane {plane_index + 1}")
    ax2d.set_xlabel('x [cm]')
    ax2d.set_ylabel('y [cm]')
    plt.colorbar(im, ax=ax2d, label='Acceptance')

    x_centers = 0.5 * (edges[:-1] + edges[1:])
    y_centers = 0.5 * (edges[:-1] + edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)
    Z = acceptance_xy.T

    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax3d.plot_surface(
        X, Y, Z,
        cmap='plasma',
        edgecolor='k',
        linewidth=0.2,
        antialiased=True
    )
    ax3d.set_title(f"3D Spatial Acceptance Surface - Plane {plane_index + 1}")
    ax3d.set_xlabel('x [cm]')
    ax3d.set_ylabel('y [cm]')
    ax3d.set_zlabel('Acceptance')
    ax3d.set_zlim(0, 1)

    plt.tight_layout()
    plt.show()