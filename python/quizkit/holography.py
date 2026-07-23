import jax
import jax.numpy as jnp
import optax

import matplotlib.pyplot as plt
import numpy as np

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
    
    U_r = jnp.fft.fft2(u_k, norm="ortho")
    I_r = jnp.abs(U_r)**2
    
    I_fourier = jnp.fft.fft2(I_r, norm="ortho")
    mu_r = jnp.real(jnp.fft.ifft2(I_fourier * otf, norm="ortho")) + background
    
    return jnp.clip(mu_r, a_min=1e-6)

def create_model_stepper(optimizer, amplitude_k, psf, target_image, mask, N, background=0.0):
    otf = jnp.fft.fft2(jnp.fft.ifftshift(psf), norm="ortho")
    trap_idx = jnp.where(mask > 0)
    
    target_traps = target_image[trap_idx]
    mask_weights = mask[trap_idx]
    
    def nll(phi_bz):
        # 1. Tile the Brillouin Zone phase patch to cover the full N x N SLM
        bz_h, bz_w = phi_bz.shape
        reps_h = (N + bz_h - 1) // bz_h
        reps_w = (N + bz_w - 1) // bz_w
        
        # Tile and crop exactly to N x N
        phi_full = jnp.tile(phi_bz, (reps_h, reps_w))[:N, :N]
        
        # 2. Forward pass uses the full tiled array
        mu_r = forward(phi_full, amplitude_k, otf, background)        
        mu_traps = mu_r[trap_idx]
        
        trap_loss = mu_traps - target_traps * jnp.log(mu_traps)
        
        return jnp.sum(mask_weights * trap_loss)
    
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

def plot_holography(phi, source_lattice, sampled_lattice, inferred_lattice):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    })
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    
    panels = [
        (axs[0, 0], phi, r'Phase Pattern', 'twilight', 0, 2*np.pi),
        (axs[0, 1], source_lattice, 'Source', 'magma', None, None),
        (axs[1, 0], sampled_lattice, 'Sampled', 'magma', None, None),
        (axs[1, 1], inferred_lattice, 'Inferred', 'magma', None, None),
    ]
    
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    return fig, axs
