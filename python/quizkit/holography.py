import jax
import jax.numpy as jnp
import h5py
import optax
import itertools
import matplotlib.pyplot as plt
import numpy as np


def get_psf(N, sigma=1.5):
    x = jnp.linspace(-N // 2, N // 2 - 1, N)
    y = jnp.linspace(-N // 2, N // 2 - 1, N)
    X, Y = jnp.meshgrid(x, y)
    psf = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / jnp.sum(psf)

    return psf


def trap_data(
    N=64,
    psf=None,
    dropout_rate=0.4,
    psf_sigma=1.5,
    source_amplitude=5_000.0,
    background_rate=1.0,
):
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    if psf is None:
        psf = get_psf(N, psf_sigma)

    spacing = 10
    grid_x, grid_y = 4, 4

    basis_vectors = jnp.array([[spacing, 0], [0, spacing]])

    extent_x = (grid_x - 1) * basis_vectors[0][0] + (grid_y - 1) * basis_vectors[1][0]
    extent_y = (grid_x - 1) * basis_vectors[0][1] + (grid_y - 1) * basis_vectors[1][1]
    
    origin = jnp.array([
        (N - extent_x) // 2,
        (N - extent_y) // 2
    ])

    trap_indices_full = list(itertools.product(range(grid_x), range(grid_y)))

    keep_mask = jax.random.bernoulli(
        subkey, p=1.0 - dropout_rate, shape=(len(trap_indices_full),)
    )
    trap_indices = [
        trap_indices_full[i] for i in range(len(trap_indices_full)) if keep_mask[i]
    ]

    mask = jnp.zeros((N, N))
    for i, j in trap_indices:
        pos = origin + i * basis_vectors[0] + j * basis_vectors[1]
        mask = mask.at[pos[0], pos[1]].set(1.0)

    xs = [int(origin[0] + i * basis_vectors[0][0] + j * basis_vectors[1][0]) for i, j in trap_indices_full]
    ys = [int(origin[1] + i * basis_vectors[0][1] + j * basis_vectors[1][1]) for i, j in trap_indices_full]
    
    pad = spacing // 2
    min_x, max_x = max(0, min(xs) - pad), min(N, max(xs) + pad + 1)
    min_y, max_y = max(0, min(ys) - pad), min(N, max(ys) + pad + 1)
    
    perimeter_mask = jnp.zeros((N, N)).at[min_x:max_x, min_y:max_y].set(1.0)

    source_lattice = source_amplitude * mask

    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    source_fourier = jnp.fft.fft2(source_lattice, norm="ortho")

    convolved_lattice = jnp.real(jnp.fft.ifft2(source_fourier * otf, norm="ortho"))
    convolved_lattice *= perimeter_mask

    expected_counts = convolved_lattice + background_rate
    sampled_lattice = jax.random.poisson(key, expected_counts).astype(jnp.float32)

    return {
        "N": N,
        "dropout_rate": dropout_rate,
        "psf_sigma": psf_sigma,
        "background_rate": background_rate,
        "source_amplitude": source_amplitude,
        "amplitude_k": jnp.ones((N, N)),
        "psf": psf,
        "source_image": convolved_lattice,
        "target_image": sampled_lattice,
        "mask": mask,
        "perimeter_mask": perimeter_mask,
        "lattice_geometry": {
            "origin": origin,
            "basis_vectors": basis_vectors,
            "trap_indices": trap_indices,
            "trap_indices_full": trap_indices_full,
        },
    }


def get_reciprocal_lattice(N, basis_vectors):
    A = np.array(basis_vectors, dtype=float)

    A_inv = np.linalg.inv(A)
    B = N * A_inv.T

    bz_shape = (
        int(np.round(np.linalg.norm(B[0]))),
        int(np.round(np.linalg.norm(B[1]))),
    )

    return B, bz_shape

def tile_image(image, N):
    h, w = image.shape
    
    # NB: If h == N and w == N, reps evaluate to 1.
    # No explicit branching is required for full-field vs BZ modes.
    reps_h = (N + h - 1) // h
    reps_w = (N + w - 1) // w
    
    tiled = jnp.tile(image, (reps_h, reps_w))
    
    return tiled[:N, :N]


def forward(N, phi, amplitude_k, psf_sigma, background):
    psf = get_psf(N, psf_sigma)   
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")

    u_k = amplitude_k * jnp.exp(1j * phi)

    u_k_shifted = jnp.fft.ifftshift(u_k)
    U_r_unshifted = jnp.fft.fft2(u_k_shifted, norm="ortho")
    U_r = jnp.fft.fftshift(U_r_unshifted)

    I_r = jnp.abs(U_r) ** 2

    I_r_shifted = jnp.fft.ifftshift(I_r)

    I_fourier = jnp.fft.fft2(I_r_shifted, norm="ortho")

    mu_r_unshifted = jnp.fft.ifft2(I_fourier * otf, norm="ortho")

    mu_r = jnp.real(jnp.fft.fftshift(mu_r_unshifted))

    return background + jnp.clip(mu_r, a_min=1e-6)


def create_model_stepper(
    optimizer, amplitude_k, target_image, N, lambda_uniformity=1.e1, lambda_sym=0.0, 
):
    y, x = jnp.ogrid[-N//2 : N//2, -N//2 : N//2]
    k2 = x**2 + y**2

    # TODO
    max_k2 = (N // 2) ** 2

    low_k_mask = (k2 <= max_k2).astype(jnp.float32)
    
    def nll(params):
        phi = params['phi']
        psf_sigma = jnp.exp(params['log_sigma'])
        background = jnp.exp(params['log_background'])
 
        phi_full = tile_image(phi, N)
     
        mu_r = forward(N, phi_full, amplitude_k, psf_sigma, background)

        loss_nll = jnp.sum(mu_r - target_image * jnp.log(mu_r + 1e-8))

        is_active = (mu_r > 0.0).astype(jnp.float32)
        n_active = jnp.sum(is_active) + 1e-8
        
        log_mu = jnp.log(mu_r + 1e-8)
        
        mean_log_active = jnp.sum(log_mu * is_active) / n_active
        variance_active = jnp.sum(is_active * (log_mu - mean_log_active)**2) / n_active
        
        uniformity_reg = lambda_uniformity * variance_active

        phasor = jnp.exp(1j * phi)
        phasor_sym = 0.25 * (
            phasor + 
            jnp.rot90(phasor, k=1) + 
            jnp.rot90(phasor, k=2) + 
            jnp.rot90(phasor, k=3)
        )
        
        sym_reg = lambda_sym * jnp.sum(jnp.abs(phasor - phasor_sym)**2)

        return loss_nll + uniformity_reg + sym_reg

    loss_and_grad_fn = jax.value_and_grad(nll)

    @jax.jit
    def stepper(params, opt_state):
        loss, grads = loss_and_grad_fn(params)        
        grads['phi'] = grads['phi'] * low_k_mask

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params_next = optax.apply_updates(params, updates)

        params_next['phi'] = jnp.mod(params_next['phi'] + jnp.pi, 2 * jnp.pi) - jnp.pi    

        return params_next, opt_state, loss

    return stepper


def write_trap_data_to_hdf5(filepath, trap_data):
    with h5py.File(filepath, "w") as f:
        f.create_dataset("source_image", data=np.array(trap_data["source_image"]))
        f.create_dataset("target_image", data=np.array(trap_data["target_image"]))
        f.create_dataset("slm_illumination", data=np.array(trap_data["amplitude_k"]))

        geom = f.create_group("lattice_geometry")
        geom.attrs["origin"] = np.array(trap_data["lattice_geometry"]["origin"])
        geom.attrs["basis_vectors"] = np.array(
            trap_data["lattice_geometry"]["basis_vectors"]
        )


def write_results_to_hdf5(filepath, trap_data, final_phi, final_inferred_intensity):
    with h5py.File(filepath, "w") as f:
        f.create_dataset("slm_phases", data=np.array(final_phi))
        f.create_dataset("inferred_lattice", data=np.array(final_inferred_intensity))

        f.create_dataset("source_image", data=np.array(trap_data["source_image"]))
        f.create_dataset("target_image", data=np.array(trap_data["target_image"]))
        f.create_dataset("slm_illumination", data=np.array(trap_data["amplitude_k"]))


def plot_holography(hdf5_path, name):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 14,
            "figure.titlesize": 16,
        }
    )

    with h5py.File(hdf5_path, "r") as f:
        source = f["source_image"][:]
        sampled = f["target_image"][:]
        phi = f["slm_phases"][:]
        inferred = f["inferred_lattice"][:]

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"{name}")

    panels = [
        (axs[0, 0], source, "source", "magma", None, None),
        (axs[0, 1], sampled, "sampled", "magma", None, None),
        (axs[1, 0], phi, r"inferred phase", "twilight", 0, 2 * np.pi),
        (axs[1, 1], inferred, "inferred intensity", "magma", None, None),
    ]

    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig, axs


def main():
    mode = "full" # Options: "bz" or "full"
    
    data = trap_data()
    N = data["N"]

    _, bz_shape = get_reciprocal_lattice(N, data["lattice_geometry"]["basis_vectors"])

    key = jax.random.PRNGKey(99)
    
    if mode == "bz":
        phi_shape = bz_shape
    else:
        phi_shape = (N, N)

    initial_params = {
        'phi': jax.random.uniform(key, phi_shape, minval=-jnp.pi, maxval=jnp.pi),
        'log_sigma': jnp.log(1.5),
        'log_background': jnp.log(1.0)
    }

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(initial_params)
    params = initial_params

    stepper = create_model_stepper(
        optimizer,
        data["amplitude_k"],
        data["target_image"],
        N,
    )

    iterations = 10_000

    print(f"Optimizing over {iterations} iterations in {mode} mode...")

    for i in range(iterations):
        params, opt_state, loss = stepper(params, opt_state)

        if i % 100 == 0 or i == iterations - 1:
            print(f"Iteration {i:04d} | Loss: {loss:.4f}")

    final_phi_full = tile_image(params['phi'], N)
    
    final_sigma = jnp.exp(params['log_sigma'])

    # NB no background.
    final_inferred_intensity = forward(
        N, final_phi_full, data["amplitude_k"], final_sigma, 0.0
    )

    write_trap_data_to_hdf5("./results/data/trap_inputs.h5", data)
    write_results_to_hdf5("./results/data/trap_results.h5", data, final_phi_full, final_inferred_intensity)

    fig, _ = plot_holography("./results/data/trap_results.h5", f"{mode.upper()}-descent phase")
    fig.savefig("./results/plots/holography.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
