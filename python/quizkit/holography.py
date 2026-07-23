import jax
import jax.numpy as jnp
import h5py
import optax
import pytest
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
        
    # Build the full perimeter mask without dropout (all 8x8 indices)
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
    
    # Center the optical axis before/after FFT
    u_k_shifted = jnp.fft.ifftshift(u_k)
    U_r_unshifted = jnp.fft.fft2(u_k_shifted, norm="ortho")
    U_r = jnp.fft.fftshift(U_r_unshifted)
    
    I_r = jnp.abs(U_r)**2
    
    # Shift back for OTF convolution
    I_r_shifted = jnp.fft.ifftshift(I_r)
    I_fourier = jnp.fft.fft2(I_r_shifted, norm="ortho")
    mu_r_unshifted = jnp.fft.ifft2(I_fourier * otf, norm="ortho")
    mu_r = jnp.real(jnp.fft.fftshift(mu_r_unshifted)) + background
    
    return jnp.clip(mu_r, a_min=1e-6)

def create_model_stepper(optimizer, amplitude_k, psf, target_image, mask, N, background=0.0):
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    trap_idx = jnp.where(mask > 0)
    
    target_traps = target_image[trap_idx]
    mask_weights = mask[trap_idx]
    
    def nll(phi_bz):
        bz_h, bz_w = phi_bz.shape
        reps_h = (N + bz_h - 1) // bz_h
        reps_w = (N + bz_w - 1) // bz_w
        
        phi_full = jnp.tile(phi_bz, (reps_h, reps_w))[:N, :N]
        
        mu_r = forward(phi_full, amplitude_k, otf, background)        
        mu_traps = mu_r[trap_idx]
        
        loss = mu_traps - target_traps * jnp.log(mu_traps)

        # NB fractional variance regularization.
        log_mu_traps = jnp.log(mu_traps)

        uniformity_regularization = 1e4 * jnp.var(log_mu_traps)
        
        return jnp.sum(mask_weights * loss) + uniformity_regularization
    
    loss_and_grad_fn = jax.value_and_grad(nll)
    
    @jax.jit
    def stepper(phi_bz, opt_state):
        # 3. Gradients are computed strictly with respect to the BZ parameters
        loss, grads = loss_and_grad_fn(phi_bz)
        updates, opt_state = optimizer.update(grads, opt_state, phi_bz)
        phi_bz_next = optax.apply_updates(phi_bz, updates)
        
        # 4. Wrap phase parameters to the principal interval [-pi, pi)
        phi_bz_next = jnp.mod(phi_bz_next + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        return phi_bz_next, opt_state, loss

    return stepper

def write_trap_data_to_hdf5(filepath, trap_data):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('target_image', data=np.array(trap_data['target_image']))
        f.create_dataset('slm_illumination', data=np.array(trap_data['amplitude_k']))
        
        geom = f.create_group('lattice_geometry')
        geom.attrs['origin'] = np.array(trap_data['lattice_geometry']['origin'])
        geom.attrs['basis_vectors'] = np.array(trap_data['lattice_geometry']['basis_vectors'])
        
def write_results_to_hdf5(filepath, trap_data, final_phi, final_inferred_intensity):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('slm_phases', data=np.array(final_phi))
        f.create_dataset('inferred_lattice', data=np.array(final_inferred_intensity))
        
        f.create_dataset('target_image', data=np.array(trap_data['target_image']))
        f.create_dataset('slm_illumination', data=np.array(trap_data['amplitude_k']))

def plot_holography(hdf5_path, name):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    })
    
    with h5py.File(hdf5_path, 'r') as f:
        source = f['target_image'][:]
        sampled = f['target_image'][:] # Or load a specific sampled variant if saved
        phi = f['slm_phases'][:]
        inferred = f['inferred_lattice'][:]
        
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Phase Retrieval: {name}")
    
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
