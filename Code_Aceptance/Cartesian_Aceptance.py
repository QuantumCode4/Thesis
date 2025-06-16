import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_spatial_acceptance(x0, y0, theta, phi, accepted_mask, L, D, N_planes, plane_index, num_bins):
    if not (0 <= plane_index < N_planes):
        raise ValueError(f"El índice del plano debe estar entre 0 y {N_planes - 1}")

    # Propagar a ese plano
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

def plot_aceptance(x0, y0, theta, phi, accepted_mask,
    L, D, N_planes, plane_index, NUM_BINS):

    acceptance_xy, edges = compute_spatial_acceptance(
    x0, y0, theta, phi, accepted_mask,
    L, D, N_planes, plane_index, NUM_BINS
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    im = ax.imshow(
        acceptance_xy.T,
        extent=extent,
        cmap='plasma',
        interpolation='nearest',
        origin='lower'
    )
    ax.set_title(f"Aceptancia Espacial en Plano {plane_index+1} (z = {plane_index*D} cm)")
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    plt.colorbar(im, ax=ax, label='Aceptancia Relativa')
    plt.tight_layout()
    plt.show()