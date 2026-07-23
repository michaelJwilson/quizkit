import jax
import jax.numpy as jnp
import h5py
import optax
import itertools
import matplotlib.pyplot as plt
import numpy as np

# @pytest.fixture
def trap_data(psf):
    N = 64
    key = jax.random.PRNGKey(42)
    # Split the key so we have one for Poisson noise and one for dropout
    key, subkey = jax.random.split(key)
    
    spacing = 8 
    origin = jnp.array([0, 0])
    
    basis_vectors = jnp.array([
        [spacing, 0], 
        [0, spacing]
    ])
    
    trap_indices_full = list(itertools.product(range(8), range(8)))
    
    dropout = 0.2
    keep_mask = jax.random.bernoulli(subkey, p=1.0 - dropout, shape=(len(trap_indices_full),))
    trap_indices = [trap_indices_full[i] for i in range(len(trap_indices_full)) if keep_mask[i]]
    
    mask = jnp.zeros((N, N))
    for (i, j) in trap_indices:
        pos = origin + i * basis_vectors[0] + j * basis_vectors[1]
        mask = mask.at[pos[0], pos[1]].set(1.0)
        
    perimeter_mask = jnp.zeros((N, N))
    for (i, j) in trap_indices_full:
        pos = origin + i * basis_vectors[0] + j * basis_vectors[1]
        perimeter_mask = perimeter_mask.at[pos[0], pos[1]].set(1.0)
            
    source_lattice = 5_000.0 * mask
    
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    source_fourier = jnp.fft.fft2(source_lattice, norm="ortho")
    convolved_lattice = jnp.real(jnp.fft.ifft2(source_fourier * otf, norm="ortho"))
    
    background = 1.0
    expected_counts = convolved_lattice + background
    sampled_lattice = jax.random.poisson(key, expected_counts).astype(jnp.float32)
    
    return {
        'N': N,
        'amplitude_k': jnp.ones((N, N)),
        'psf': psf,
        'source_image': convolved_lattice,
        'target_image': sampled_lattice,
        'mask': mask,
        'perimeter_mask': perimeter_mask,
        'background': background,
        'lattice_geometry': {
            'origin': origin,
            'basis_vectors': basis_vectors,
            'trap_indices': trap_indices,
            'trap_indices_full': trap_indices_full
        }
    }

def get_reciprocal_lattice(N, basis_vectors):
    A = np.array(basis_vectors, dtype=float)
    
    A_inv = np.linalg.inv(A)
    B = N * A_inv.T

    bz_shape = (
        int(np.round(np.linalg.norm(B[0]))),
        int(np.round(np.linalg.norm(B[1])))
    )
    
    return B, bz_shape

def forward(phi, amplitude_k, otf, background):
    u_k = amplitude_k * jnp.exp(1j * phi)
    
    u_k_shifted = jnp.fft.ifftshift(u_k)
    U_r_unshifted = jnp.fft.fft2(u_k_shifted, norm="ortho")
    U_r = jnp.fft.fftshift(U_r_unshifted)
    
    I_r = jnp.abs(U_r)**2

    I_r_shifted = jnp.fft.ifftshift(I_r)
    I_fourier = jnp.fft.fft2(I_r_shifted, norm="ortho")
    mu_r_unshifted = jnp.fft.ifft2(I_fourier * otf, norm="ortho")
    mu_r = jnp.real(jnp.fft.fftshift(mu_r_unshifted)) + background
    
    return jnp.clip(mu_r, a_min=1e-6)

def create_model_stepper(
    optimizer, 
    amplitude_k, 
    psf, 
    target_image, 
    known_mask, 
    N, 
    background=1.0
):
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    
    # 1. Dynamic Active Trap Identification
    # We do not know the dropout mask a priori, so we infer it.
    # We filter the known_mask for locations where the target image 
    # is distinctly above the background noise floor.
    signal_threshold = 2.0 * background 
    active_trap_idx = jnp.where((known_mask > 0) & (target_image > signal_threshold))
    
    def nll(phi_bz):
        bz_h, bz_w = phi_bz.shape
        reps_h = (N + bz_h - 1) // bz_h
        reps_w = (N + bz_w - 1) // bz_w
        phi_full = jnp.tile(phi_bz, (reps_h, reps_w))[:N, :N]
        
        mu_r = forward(phi_full, amplitude_k, otf, background) 
        
        # 2. Full-Image Data Fidelity
        # We evaluate NLL over the *entire* image, not just the mask. 
        # This naturally forces mu_r to match the background level at 
        # dropout locations within the known_mask.
        loss_nll = jnp.sum(mu_r - target_image * jnp.log(mu_r + 1e-8))
        
        # 3. Targeted Uniformity Regularization
        # We calculate variance ONLY on the traps we inferred are active.
        mu_active = mu_r[active_trap_idx]
        log_mu_active = jnp.log(mu_active + 1e-8)
        uniformity_reg = 7e4 * jnp.var(log_mu_active)
        
        return loss_nll + uniformity_reg

    loss_and_grad_fn = jax.value_and_grad(nll)

    @jax.jit
    def stepper(phi_bz, opt_state):
        loss, grads = loss_and_grad_fn(phi_bz)
        updates, opt_state = optimizer.update(grads, opt_state, phi_bz)
        phi_bz_next = optax.apply_updates(phi_bz, updates)
        
        # Wrap phase parameters to the principal interval [-pi, pi)
        phi_bz_next = jnp.mod(phi_bz_next + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        return phi_bz_next, opt_state, loss

    return stepper

def write_trap_data_to_hdf5(filepath, trap_data):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('source_image', data=np.array(trap_data['source_image']))
        f.create_dataset('target_image', data=np.array(trap_data['target_image']))
        f.create_dataset('slm_illumination', data=np.array(trap_data['amplitude_k']))
        
        geom = f.create_group('lattice_geometry')
        geom.attrs['origin'] = np.array(trap_data['lattice_geometry']['origin'])
        geom.attrs['basis_vectors'] = np.array(trap_data['lattice_geometry']['basis_vectors'])
        
def write_results_to_hdf5(filepath, trap_data, final_phi, final_inferred_intensity):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('slm_phases', data=np.array(final_phi))
        f.create_dataset('inferred_lattice', data=np.array(final_inferred_intensity))

        # TODO
        f.create_dataset('source_image', data=np.array(trap_data['source_image']))
        f.create_dataset('target_image', data=np.array(trap_data['target_image']))
        f.create_dataset('slm_illumination', data=np.array(trap_data['amplitude_k']))

def plot_holography(hdf5_path, name):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    })
    
    with h5py.File(hdf5_path, 'r') as f:
        source = f['source_image'][:]
        sampled = f['target_image'][:] # Or load a specific sampled variant if saved
        phi = f['slm_phases'][:]
        inferred = f['inferred_lattice'][:]
        
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"{name}")
    
    panels = [
        (axs[0, 0], source, 'Source', 'magma', None, None),
        (axs[0, 1], sampled, 'Sampled', 'magma', None, None),
        (axs[1, 0], phi, r'Inferred Phase', 'twilight', 0, 2*np.pi),
        (axs[1, 1], inferred, 'Inferred Intensity', 'magma', None, None),
    ]
    
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    return fig, axs

def get_psf(N, sigma=1.5):
    x = jnp.linspace(-N//2, N//2 - 1, N)
    y = jnp.linspace(-N//2, N//2 - 1, N)
    X, Y = jnp.meshgrid(x, y)
    psf = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / jnp.sum(psf)

    return psf


def main():
    N = 64
    sigma = 1.5

    psf = get_psf(N, sigma)
    data = trap_data(psf)

    _, bz_shape = get_reciprocal_lattice(N, data['lattice_geometry']['basis_vectors'])

    print(f"Calculated BZ shape: {bz_shape}")

    key = jax.random.PRNGKey(99)
    phi_bz = jax.random.uniform(key, bz_shape, minval=-jnp.pi, maxval=jnp.pi)

    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(phi_bz)

    stepper = create_model_stepper(
        optimizer, 
        data['amplitude_k'], 
        data['psf'], 
        data['target_image'], 
        data['perimeter_mask'],
        data['N'], 
        data['background']
    )

    iterations = 1_000

    print(f"Optimizing over {iterations} iterations...")

    for i in range(iterations):
        phi_bz, opt_state, loss = stepper(phi_bz, opt_state)

        if i % 100 == 0 or i == iterations - 1:
            print(f"Iteration {i:04d} | Loss: {loss:.4f}")

    bz_h, bz_w = phi_bz.shape
    reps_h = (N + bz_h - 1) // bz_h
    reps_w = (N + bz_w - 1) // bz_w
    final_phi_full = jnp.tile(phi_bz, (reps_h, reps_w))[:N, :N]

    otf = jnp.fft.fft2(jnp.fft.ifftshift(data['psf']), norm="ortho")
    final_inferred_intensity = forward(
        final_phi_full, 
        data['amplitude_k'], 
        otf, 
        data['background']
    )

    input_hdf5 = "trap_inputs.h5"
    results_hdf5 = "trap_results.h5"
    
    print(f"Writing inputs to {input_hdf5}...")
    write_trap_data_to_hdf5(input_hdf5, data)
    
    print(f"Writing results to {results_hdf5}...")
    write_results_to_hdf5(results_hdf5, data, final_phi_full, final_inferred_intensity)

    print("Generating plots...")
    fig, axs = plot_holography(results_hdf5, "BZ-descent phase")
    
    plt.savefig("holography.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()