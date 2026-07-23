import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable

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

        uniformity_regularization = 1e3 * jnp.var(log_mu_traps)
        
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

def plot_holography(phi, source_lattice, sampled_lattice, inferred_lattice):
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    })
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    
    panels = [
        (axs[0, 0], phi, r'Phase', 'twilight', 0, 2*np.pi),
        (axs[0, 1], source_lattice, 'Source', 'magma', None, None),
        (axs[1, 0], sampled_lattice, 'Sampled', 'magma', None, None),
        (axs[1, 1], inferred_lattice, 'Inferred', 'magma', None, None),
    ]
    
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        
        fig.colorbar(im, cax=cax)
        
    return fig, axs
