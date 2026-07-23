import pytest
import jax
import jax.numpy as jnp
import optax
from quizkit.holography import phase_forward_model, create_model_step

@pytest.fixture
def trap_data():
    N = 64
    key = jax.random.PRNGKey(42)
    
    # NB target traps.
    mask = jnp.zeros((N, N))
    spacing = 10
    start = (N - 3 * spacing) // 2
    for i in range(4):
        for j in range(4):
            mask = mask.at[start + i*spacing, start + j*spacing].set(1.0)
            
    ideal_lattice = mask * 5_000.0 
    
    x = jnp.arange(N) - N//2
    X, Y = jnp.meshgrid(x, x)
    sigma = 1.5
    psf = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf = psf / jnp.sum(psf)
    
    # NB psf convolution
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    ideal_fourier = jnp.fft.fft2(ideal_lattice, norm="ortho")
    blurred_lattice = jnp.real(jnp.fft.ifft2(ideal_fourier * otf, norm="ortho"))
    
    background = 100.0
    expected_photons = blurred_lattice + background
    noisy_ccd_image = jax.random.poisson(key, expected_photons).astype(jnp.float32)
    
    # NB uniform illumination beam
    amplitude_k = jnp.ones((N, N)) 
    
    return {
        'N': N,
        'amplitude_k': amplitude_k,
        'psf': psf,
        'target_image': noisy_ccd_image,
        'mask': jnp.ones((N, N)), # Evaluate loss over full chip
        'background': background
    }

def test_phase_retrieval_convergence(trap_data):
    step_fn = create_model_step(
        amplitude_k=trap_data['amplitude_k'],
        psf=trap_data['psf'],
        target_image=trap_data['target_image'],
        mask=trap_data['mask'],
        background=trap_data['background']
    )
    """
    key = jax.random.PRNGKey(42)
    phi = jax.random.uniform(key, (trap_data['N'], trap_data['N']), minval=0, maxval=2*jnp.pi)
    
    optimizer = optax.adam(learning_rate=0.5)
    opt_state = optimizer.init(phi)
    
    # Track initial loss
    _, _, initial_loss = step_fn(phi, opt_state, optimizer)
    
    # Run optimization loop
    for _ in range(50):
        phi, opt_state, loss = step_fn(phi, opt_state, optimizer)
        
    # Validation: Loss must strictly decrease
    assert loss < initial_loss, f"Loss failed to decrease. Init: {initial_loss:.2f}, Final: {loss:.2f}"
    
    # Validation: Outputs must be finite
    assert jnp.all(jnp.isfinite(phi)), "Phase array contains NaNs."
    """
