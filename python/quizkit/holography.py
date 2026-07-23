import jax
import jax.numpy as jnp
import optax

def forward(phi, amplitude_k, otf, background):
    u_k = amplitude_k * jnp.exp(1j * phi)
    
    U_r = jnp.fft.fft2(u_k, norm="ortho")
    I_r = jnp.abs(U_r)**2
    
    I_fourier = jnp.fft.fft2(I_r, norm="ortho")
    mu_r = jnp.real(jnp.fft.ifft2(I_fourier * otf, norm="ortho")) + background
    
    return jnp.clip(mu_r, a_min=1e-6)

def create_model_stepper(optimizer, amplitude_k, psf, target_image, mask, background=0.0):
    # Precompute OTF once
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    
    # 1. Precompute indices: This executes eagerly in Python just once.
    # jnp.where returns a tuple of arrays (row_indices, col_indices)
    trap_idx = jnp.where(mask > 0)
    
    # 2. Extract static target values and weights at those coordinates
    target_traps = target_image[trap_idx]
    mask_weights = mask[trap_idx]
    
    def nll(phi):
        # The FFTs still process the full NxN array
        mu_r = forward(phi, amplitude_k, otf, background)
        
        # 3. Gather: Extract only the predicted intensities at the trap locations
        mu_traps = mu_r[trap_idx]
        
        # 4. Compute the loss ONLY on the active subset
        trap_loss = mu_traps - target_traps * jnp.log(mu_traps)
        
        return jnp.sum(mask_weights * trap_loss)
    
    loss_and_grad_fn = jax.value_and_grad(nll)
    
    @jax.jit
    def stepper(phi, opt_state):
        loss, grads = loss_and_grad_fn(phi)
        updates, opt_state = optimizer.update(grads, opt_state, phi)
        phi_next = optax.apply_updates(phi, updates)
        return phi_next, opt_state, loss

    return stepper
