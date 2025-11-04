import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def _calculate_spectrum_row(g1_row, tau_array, omega_array):
    """
    (Internal JIT-compiled function)
    Calculates the spectrum S(w) for a *single* time-slice (one row of G1).
    
    S(w) = Re[ integral( G1(tau) * exp(i*w*tau) d(tau) ) ]
    
    Args:
        g1_row (jax.Array): The 1D array of G1(tau) for a fixed t. Shape (N_tau,).
        tau_array (jax.Array): The full 1D array of tau values. Shape (N_tau,).
        omega_array (jax.Array): The 1D array of frequencies (w) to calculate. Shape (N_omega,).
        
    Returns:
        jax.Array: The 1D spectrum S(w) for the given t. Shape (N_omega,).
    """
    
    # 1. Create the exp(i*w*tau) matrix
    # We use 'None' to broadcast the 1D arrays to 2D
    omega_col = omega_array[:, None] # Shape (N_omega, 1)
    tau_row = tau_array[None, :]     # Shape (1, N_tau)
    
    # exp_matrix[j, k] = exp(i * omega_j * tau_k)
    exp_matrix = jnp.exp(1j * omega_col * tau_row) # Shape (N_omega, N_tau)

    # 2. Calculate the integrand
    # g1_row shape (N_tau,) is broadcast against (N_omega, N_tau)
    integrand_matrix = exp_matrix * g1_row

    # 3. Integrate over the tau axis (axis=1) using the trapezoidal rule
    # This is the "Riemann-like sum" you requested.
    integral_vector = jnp.trapezoid(integrand_matrix, x=tau_array, axis=1)

    # 4. Return the real part (as per your definition)
    S_row = jnp.real(integral_vector)
    return S_row

def calculate_spectrum_matrix(g1_matrix, tau_array, omega_array):
    """
    Calculates the time-dependent emission spectrum S(t, w) from G1(t, tau).
    
    Performs the Fourier transform integral for each time-slice 't' using
    a vectorized trapezoidal sum (via jax.vmap).
    
    Definition:
    S(t, w) = Re[ integral( G1(t, tau) * exp(i*w*tau) d(tau) ) ]
    
    Args:
        g1_matrix (jax.Array): The 2D matrix of G1(t, tau). 
                               Shape (N_t, N_tau_full).
        tau_array (jax.Array): The full 1D array of tau values (neg to pos).
                               Shape (N_tau_full,).
        omega_array (jax.Array): The 1D array of frequencies (w) to calculate.
                                 Shape (N_omega,).
                                 
    Returns:
        jax.Array: The 2D spectrum matrix S(t, w). Shape (N_t, N_omega).
    """
    print("Calculating spectrum matrix S(t, w)...")
    
    # We vmap the row-calculation function over the g1_matrix (axis 0)
    # in_axes=(0, None, None) means:
    #   - Iterate over g1_matrix (axis 0)
    #   - Broadcast tau_array (use the same array for all rows)
    #   - Broadcast omega_array (use the same array for all rows)
    S_matrix = jax.vmap(_calculate_spectrum_row, in_axes=(0, None, None))(
        g1_matrix, tau_array, omega_array
    )
    
    print("Spectrum calculation complete.")
    return S_matrix

