import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import optax

def phase_forward_model(phi, amplitude_k, otf, background):
    """Core physics forward pass with energy conservation."""
    # 1. SLM Field
    u_k = amplitude_k * jnp.exp(1j * phi)
    
    # 2. Focal plane intensity (conserving energy via norm="ortho")
    U_r = jnp.fft.fft2(u_k, norm="ortho")
    I_r = jnp.abs(U_r)**2
    
    # 3. PSF Convolution
    I_fourier = jnp.fft.fft2(I_r, norm="ortho")
    mu_r = jnp.real(jnp.fft.ifft2(I_fourier * otf, norm="ortho")) + background
    
    return jnp.clip(mu_r, a_min=1e-6)

def poisson_nll(phi, amplitude_k, otf, background, target_image, mask):
    """Exact Poisson negative log-likelihood."""
    mu_r = phase_forward_model(phi, amplitude_k, otf, background)
    
    # Poisson NLL: mu - Y * log(mu). 
    # Constant terms dependent only on Y are dropped.
    pixel_loss = mu_r - target_image * jnp.log(mu_r)
    weighted_loss = mask * pixel_loss
    
    return jnp.sum(weighted_loss)

def create_model_step(amplitude_k, psf, target_image, mask, background=0.0):
    """Factory returns a JIT-compiled update step function."""
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    
    # Freeze the static parameters into the loss function
    loss_fn = Partial(
        poisson_nll, 
        amplitude_k=amplitude_k, 
        otf=otf, 
        background=background,
        target_image=target_image, 
        mask=mask
    )
    
    loss_and_grad_fn = jax.value_and_grad(loss_fn)
    
    @jax.jit
    def step_fn(phi, opt_state, optimizer):
        loss, grads = loss_and_grad_fn(phi)
        updates, opt_state = optimizer.update(grads, opt_state, phi)
        phi_next = optax.apply_updates(phi, updates)
        return phi_next, opt_state, loss

    return step_fn