import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
import numpy as np

def forward(phi, amplitude_k, otf, background):
    u_k = amplitude_k * jnp.exp(1j * phi)
    
    U_r = jnp.fft.fft2(u_k, norm="ortho")
    I_r = jnp.abs(U_r)**2
    
    I_fourier = jnp.fft.fft2(I_r, norm="ortho")
    mu_r = jnp.real(jnp.fft.ifft2(I_fourier * otf, norm="ortho")) + background
    
    return jnp.clip(mu_r, a_min=1e-6)

def create_model_stepper(optimizer, amplitude_k, psf, target_image, mask, background=0.0):
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    trap_idx = jnp.where(mask > 0)
    
    target_traps = target_image[trap_idx]
    mask_weights = mask[trap_idx]
    
    def nll(phi):
        mu_r = forward(phi, amplitude_k, otf, background)        
        mu_traps = mu_r[trap_idx]
        
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

def plot_holography(phi, source_lattice, sampled_lattice, inferred_lattice):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    })
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    
    panels = [
        (axs[0, 0], phi, r'Inferred Phase Pattern ($\phi$)', 'twilight', 0, 2*np.pi),
        (axs[0, 1], source_lattice, 'Source Lattice (Ideal)', 'magma', None, None),
        (axs[1, 0], sampled_lattice, 'Sampled Lattice (Target)', 'magma', None, None),
        (axs[1, 1], inferred_lattice, 'Inferred Lattice (Prediction)', 'magma', None, None),
    ]
    
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    return fig, axs
