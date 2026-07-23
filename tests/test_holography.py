import pytest
import jax
import jax.numpy as jnp
import optax
import itertools
from quizkit.holography import forward, create_model_stepper, get_reciprocal_lattice, plot_holography

@pytest.fixture
def psf(N=64, sigma=1.5):
    x = jnp.arange(N) - N//2
    X, Y = jnp.meshgrid(x, x)
    psf = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return psf / jnp.sum(psf)

@pytest.fixture
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
    
    # 1. Full 8x8 grid
    trap_indices_full = list(itertools.product(range(8), range(8)))
    
    # 2. Apply 20% dropout
    dropout = 0.2
    keep_mask = jax.random.bernoulli(subkey, p=1.0 - dropout, shape=(len(trap_indices_full),))
    trap_indices = [trap_indices_full[i] for i in range(len(trap_indices_full)) if keep_mask[i]]
    
    mask = jnp.zeros((N, N))
    for (i, j) in trap_indices:
        pos = origin + i * basis_vectors[0] + j * basis_vectors[1]
        mask = mask.at[pos[0], pos[1]].set(1.0)
            
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
        'background': background,
        'lattice_geometry': {
            'origin': origin,
            'basis_vectors': basis_vectors,
            'trap_indices': trap_indices
        }
    }

# NB pytest -s tests/test_holography.py::test_holography
def test_holography(trap_data):
    learning_rate = 1e-1
    optimizer = optax.adam(learning_rate=learning_rate)
    
    # 1. Added missing 'N' argument
    stepper = create_model_stepper(
        optimizer=optimizer,
        amplitude_k=trap_data['amplitude_k'],
        psf=trap_data['psf'],
        target_image=trap_data['target_image'],
        mask=trap_data['mask'],
        N=trap_data['N'],
        background=trap_data['background']
    )

    key = jax.random.PRNGKey(42)

    B, bz_shape = get_reciprocal_lattice(
        trap_data['N'], 
        trap_data['lattice_geometry']['basis_vectors']
    )

    phi_bz = jax.random.uniform(key, bz_shape, minval=-jnp.pi, maxval=jnp.pi)
    opt_state = optimizer.init(phi_bz)

    # 2. Corrected variable names to phi_bz
    _, _, initial_loss = stepper(phi_bz, opt_state)

    for _ in range(100):
        phi_bz, opt_state, loss = stepper(phi_bz, opt_state)
        
    assert loss < initial_loss, f"Loss failed to decrease. Init: {initial_loss:.2f}, Final: {loss:.2f}"
    assert jnp.all(jnp.isfinite(phi_bz)), "Phase array contains NaNs."

    otf = jnp.fft.fft2(jnp.fft.ifftshift(trap_data['psf']), norm="ortho")
    
    bz_h, bz_w = phi_bz.shape
    reps_h = (trap_data['N'] + bz_h - 1) // bz_h
    reps_w = (trap_data['N'] + bz_w - 1) // bz_w
    phi_full = jnp.tile(phi_bz, (reps_h, reps_w))[:trap_data['N'], :trap_data['N']]
    
    inferred_lattice = forward(
        phi_full, trap_data['amplitude_k'], otf, trap_data['background']
    )

    inferred_masked = inferred_lattice * trap_data['mask']
    
    source_lattice = 5_000.0 * trap_data['mask']
    source_fourier = jnp.fft.fft2(source_lattice, norm="ortho")
    convolved_lattice = jnp.real(jnp.fft.ifft2(source_fourier * otf, norm="ortho")) + trap_data['background']

    phi_plot = jnp.mod(phi_full, 2 * jnp.pi)

    fig, _ = plot_holography(
        phi=phi_plot, 
        source_lattice=convolved_lattice, 
        sampled_lattice=trap_data['target_image'], 
        inferred_lattice=inferred_masked  # Pass the masked version
    )
    
    fig.savefig('./holography.pdf', dpi=300, bbox_inches='tight')
