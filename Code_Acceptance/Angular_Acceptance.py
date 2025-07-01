import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import interp1d

def sample_theta_cosn(N, n, method):
    if method == "rejection":
        samples = []
        max_batch = 10_000_000

        with tqdm(total=N, desc=f"Sampling θ with cos^{n}(θ) [rejection]") as pbar:
            while len(samples) < N:
                remaining = N - len(samples)
                n_batch = min(max_batch, int(remaining * 1.5))

                theta_candidates = np.random.uniform(0, np.pi/2, size=n_batch)
                probs = np.cos(theta_candidates) ** n

                uniform_randoms = np.random.uniform(0, 1, size=n_batch)
                accepted = theta_candidates[uniform_randoms < probs]

                accepted = accepted[:remaining]
                samples.extend(accepted.tolist())
                pbar.update(len(accepted))

        return np.array(samples)

    elif method == "cumulative":
        print(f"Sampling θ with cos^{n}(θ) [cumulative]")
        N_table = 10000

        theta_vals = np.linspace(0, np.pi/2, N_table)
        pdf_vals = np.cos(theta_vals) ** n

        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals /= cdf_vals[-1]

        inverse_cdf = interp1d(cdf_vals, theta_vals, kind='linear', bounds_error=False, fill_value=(0, np.pi/2))
        u = np.random.uniform(0, 1, N)
        samples = inverse_cdf(u)

        return np.array(samples)

    else:
        raise ValueError("El parámetro 'method' debe ser 'cumulative' o 'rejection'")

def generate_muons(N_muons, L, D, N_planes, n, method):
    phi = np.random.uniform(0, 2 * np.pi, N_muons)
    theta = sample_theta_cosn(N_muons,n,method)
    tan_theta = np.tan(theta)

    x0 = np.random.uniform(-L/2, L/2, N_muons)
    y0 = np.random.uniform(-L/2, L/2, N_muons)

    accepted_mask = np.ones(N_muons, dtype=bool)

    with tqdm(total=N_planes-1, desc="Simulating Trajectories") as pbar:
        for i in range(1, N_planes):
            x = x0 + i * D * tan_theta * np.cos(phi)
            y = y0 + i * D * tan_theta * np.sin(phi)
            accepted_mask &= (np.abs(x) <= L/2) & (np.abs(y) <= L/2)
            pbar.update(1)

    return x0, y0, theta, phi, accepted_mask

def plot_aceptance(theta, accepted_theta, theta_max, NUM_BINS):

    counts_accepted, bin_edges = np.histogram(accepted_theta, bins=NUM_BINS, range=(0, np.pi/2))
    counts_generated, _ = np.histogram(theta, bins=NUM_BINS, range=(0, np.pi/2))

    prob_acceptance_per_bin = np.divide(
        counts_accepted,
        counts_generated,
        out=np.zeros_like(counts_accepted, dtype=float),
        where=counts_generated != 0
    )

    bin_centers_deg = np.rad2deg((bin_edges[:-1] + bin_edges[1:]) / 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_centers_deg, prob_acceptance_per_bin,
       width=np.rad2deg(bin_edges[1]-bin_edges[0]),
       edgecolor='black')

    ax.set_title('Hodoscope Acceptance')
    ax.set_xlabel('Angle θ [grades]')
    ax.set_ylabel('Acceptance')
    ax.grid(True)
    ax.set_xlim(0, 90)
    ax.set_ylim(0)
    plt.axvline(np.rad2deg(theta_max), color='k', linestyle='--', label='θ_max')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_theta_xy(theta, phi, accepted_mask, N_planes, D, L, NUM_BINS):

    theta_x = np.rad2deg(np.arctan(np.tan(theta) * np.cos(phi)))
    theta_y = np.rad2deg(np.arctan(np.tan(theta) * np.sin(phi)))

    accepted_theta_x = theta_x[accepted_mask]
    accepted_theta_y = theta_y[accepted_mask]

    theta_max1 = np.arctan(L / ((N_planes - 1) * D))
    bin_edges_2d = np.linspace(-np.rad2deg(theta_max1), np.rad2deg(theta_max1), NUM_BINS)

    H_gen, _, _ = np.histogram2d(theta_x, theta_y, bins=(bin_edges_2d, bin_edges_2d))
    H_acc, _, _ = np.histogram2d(accepted_theta_x, accepted_theta_y, bins=(bin_edges_2d, bin_edges_2d))

    acceptance_2D = np.divide(
        H_acc,
        H_gen,
        out=np.zeros_like(H_acc),
        where=H_gen != 0
    )

    fig = plt.figure(figsize=(12, 6))
    extent = [bin_edges_2d[0], bin_edges_2d[-1], bin_edges_2d[0], bin_edges_2d[-1]]


    theta_x_centers = 0.5 * (bin_edges_2d[:-1] + bin_edges_2d[1:])
    theta_y_centers = 0.5 * (bin_edges_2d[:-1] + bin_edges_2d[1:])
    X, Y = np.meshgrid(theta_x_centers, theta_y_centers)
    Z = acceptance_2D
    
    ax2d = fig.add_subplot(1, 2, 1)
    im = ax2d.imshow(acceptance_2D.T, extent=extent, cmap='jet', interpolation='nearest',
        origin='lower')
    ax2d.set_title("Angular Acceptance Map")
    ax2d.set_xlabel(r'$\theta_x$ [grades]')
    ax2d.set_ylabel(r'$\theta_y$ [grades]')
    plt.colorbar(im, ax=ax2d, label='Acceptance')

    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax3d.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')
    ax3d.set_title("Angular Acceptance Surface")
    ax3d.set_xlabel(r'$\theta_x$ [grades]')
    ax3d.set_ylabel(r'$\theta_y$ [grades]')
    ax3d.set_zlabel("Acceptance")

    plt.tight_layout()
    plt.show()