import pytest
import jax
import jax.numpy as jnp
import optax
from quizkit.holography import create_model_stepper

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
    
    # Generate mask (16 traps)
    mask = jnp.zeros((N, N))
    spacing = 10
    start = (N - 3 * spacing) // 2
    for i in range(4):
        for j in range(4):
            mask = mask.at[start + i*spacing, start + j*spacing].set(1.0)
            
    source_lattice = 5_000.0 * mask
    
    # PSF convolution setup
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    source_fourier = jnp.fft.fft2(source_lattice, norm="ortho")
    convolved_lattice = jnp.real(jnp.fft.ifft2(source_fourier * otf, norm="ortho"))
    
    background = 100.0
    expected_counts = convolved_lattice + background
    sampled_lattice = jax.random.poisson(key, expected_counts).astype(jnp.float32)
    
    return {
        'N': N,
        'amplitude_k': jnp.ones((N, N)),
        'psf': psf,
        'target_image': sampled_lattice,
        'mask': mask,
        'background': background
    }

def test_holography(trap_data):
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate=learning_rate)
    
    # We pass the optimizer in during initialization
    stepper = create_model_stepper(
        optimizer=optimizer,
        amplitude_k=trap_data['amplitude_k'],
        psf=trap_data['psf'],
        target_image=trap_data['target_image'],
        mask=trap_data['mask'],
        background=trap_data['background']
    )

    key = jax.random.PRNGKey(42)
    phi = jax.random.uniform(key, (trap_data['N'], trap_data['N']), minval=0, maxval=2*jnp.pi)
    
    opt_state = optimizer.init(phi)
    
    # Notice `optimizer` is no longer passed on every step
    _, _, initial_loss = stepper(phi, opt_state)
    
    for _ in range(50):
        phi, opt_state, loss = stepper(phi, opt_state)
        
    assert loss < initial_loss, f"Loss failed to decrease. Init: {initial_loss:.2f}, Final: {loss:.2f}"
    assert jnp.all(jnp.isfinite(phi)), "Phase array contains NaNs."
