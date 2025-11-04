from evolution import solve_dynamics
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy.interpolate import RectBivariateSpline

@jax.jit
def calculate_trace(sigma, adag_L):
    """
    Calculates Tr(adag_L * sigma).
    
    Args:
        sigma (jax.Array): The density-like matrix.
        adag_L (jax.Array): The operator to trace against (e.g., a_dag).
    """
    # Calculates Tr(a_dag * sigma)
    return jnp.trace(adag_L @ sigma)


@partial(jax.jit, static_argnames=("H_t_func", "E_func"))
def get_g1_row(
    rho_t,          
    t,              
    a_L,            
    adag_L,         
    tau_array,      
    L_ops,          
    H_t_func,      
    E_func,         
) -> jnp.ndarray:
    """
    Calculates a single row of the G1(t, tau) matrix for a given t
    and for TAU >= 0.
    
    *** This version is corrected for time-dependent Hamiltonians. ***
    
    Args:
        rho_t (jax.Array): The density matrix at time t, rho(t).
        t (float): The absolute time t.
        a_L (jax.Array): The annihilation operator.
        adag_L (jax.Array): The creation operator.
        tau_array (jax.Array): The array of tau values (must be >= 0).
        L_ops (list of jax.Array): The list of collapse operators.
        H_t_func (function): Function returning H(t, E).
        E_func (function): Function returning E(t).
    Returns:
        jax.Array: The row of G1(t, tau) for the given t and
    """
    
    # 1. QRT initial condition: sigma(t, 0) = a * rho(t)
    sigma_0 = a_L @ rho_t
    
    # The QRT evolution from t to t+tau must use the
    # time-dependent Hamiltonian H(t + tau') and E(t + tau').
    # We create new functions that are "closed over" the absolute time t.
    
    # H_tau_func(tau, E) will call H_t_func(t + tau, E)
    H_tau_func = lambda tau, E_f: H_t_func(t + tau, E_f)
    # E_tau_func(tau) will call E_func(t + tau)
    E_tau_func = lambda tau: E_func(t + tau)
    
    # 2. Solve QRT evolution: d(sigma)/d(tau) = L(sigma)
    sigma_all_tau = solve_dynamics(
        sigma_0, 
        tau_array, 
        H_tau_func,  # Use the shifted H function
        E_tau_func,  # Use the shifted E function
        L_ops
    )
    
    # 3. Calculate G1(t, tau) = Tr(a_dag * sigma(t, tau))
    G1_row = jax.vmap(calculate_trace, in_axes=(0, None))(sigma_all_tau, adag_L)
    
    return G1_row

def g1_matrix(
    rho_t_array,    
    t_array,        
    a_L,            
    adag_L,         
    tau_array_pos,
    tau_array_neg,      
    L_ops,          
    H_t_func,      
    E_func          
) -> jnp.ndarray:
    """
    Calculates the full G1(t, tau) matrix for given arrays of t and tau.
    
    Args:
        rho_t_array (jax.Array): Array of density matrices at times t_array.
        t_array (jax.Array): Array of absolute times t.
        a_L (jax.Array): The annihilation operator.
        adag_L (jax.Array): The creation operator.
        tau_array (jax.Array): The array of tau values (must be >= 0).
        L_ops (list of jax.Array): The list of collapse operators.
        H_t_func (function): Function returning H(t, E).
        E_func (function): Function returning E(t).
    Returns:
        jnp.ndarray: The G1(t, tau) matrix.
    """

    if tau_array_neg == None:

        G1_positive_matrix_list = []
        for i in range(len(t_array)):
            # Call the imported, JIT-compiled function
            G1_row = get_g1_row(
                rho_t_array[i],   # rho_t
                t_array[i],
                a_L,            # a_L
                adag_L,         # adag_L
                tau_array_pos,      # tau_array
                L_ops,          # L_ops
                H_t_func,     # H_t_func
                E_func    # E_func
            )
            G1_positive_matrix_list.append(G1_row)

        G1_positive_matrix = jnp.array(G1_positive_matrix_list)
        return G1_positive_matrix
    
    elif tau_array_neg != None:
        G1_positive_matrix_list = []

        for i in range(len(t_array)):
            # Call the imported, JIT-compiled function
            G1_row = get_g1_row(
                rho_t_array[i],   # rho_t
                t_array[i],
                a_L,            # a_L
                adag_L,         # adag_L
                tau_array_pos,      # tau_array
                L_ops,          # L_ops
                H_t_func,     # H_t_func
                E_func    # E_func
            )
            G1_positive_matrix_list.append(G1_row)

        G1_positive_matrix = jnp.array(G1_positive_matrix_list)

        G1_real_np = np.array(G1_positive_matrix.real)
        G1_imag_np = np.array(G1_positive_matrix.imag)

        interp_real = RectBivariateSpline(
            np.array(t_array), np.array(tau_array_pos), G1_real_np, kx=1, ky=1
        )
        interp_imag = RectBivariateSpline(
            np.array(t_array), np.array(tau_array_pos), G1_imag_np, kx=1, ky=1
        )

        G1_negative_matrix_list = []

        for i in range(len(t_array)):
            t_i = t_array[i]
            G1_neg_row = []
            
            # Loop over all negative delay times 'tau_neg'
            for tau_neg in tau_array_neg:
                # These are the args for the identity: G1(t, tau_neg) = conj[G1(t', tau_d)]
                tau_d = -tau_neg      # The corresponding positive delay (e.g., 1.5)
                t_prime = t_i + tau_neg # The shifted time (e.g., t_i - 1.5)
                
                # We find G1(t', tau_d) by "looking it up" in our interpolator
                # 'grid=False' tells the interpolator we are asking for one point
                G_helper_real = interp_real(t_prime, tau_d, grid=False)
                G_helper_imag = interp_imag(t_prime, tau_d, grid=False)
                
                G_helper = G_helper_real + 1j * G_helper_imag
                
                # Apply the identity
                G_val = jnp.conj(G_helper)
                G1_neg_row.append(G_val)
                
            G1_negative_matrix_list.append(jnp.array(G1_neg_row))

        G1_negative_matrix = jnp.array(G1_negative_matrix_list)

        G1_matrix_full = jnp.hstack([G1_negative_matrix, G1_positive_matrix])
        return G1_matrix_full