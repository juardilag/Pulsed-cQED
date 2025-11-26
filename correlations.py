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
    L_ops,          
    H_t_func,      
    E_func          
) -> jnp.ndarray:
    """
    Calculates the full G1(t, tau) matrix for given arrays of t and tau.
    
    If tau_array_neg is None, calculates only for tau >= 0.
    
    If tau_array_neg is provided, calculates the full matrix using the
    "aligned grid" (no-interpolation) method. This *requires*
    that the time steps in t_array and tau_array_pos are identical.
    
    Args:
        rho_t_array (jax.Array): Array of density matrices at times t_array.
        t_array (jax.Array): Array of absolute times t.
        a_L (jax.Array): The annihilation operator (or sigma_minus).
        adag_L (jax.Array): The creation operator (or sigma_plus).
        tau_array_pos (jax.Array): Array of non-negative tau values (e.g., [0, dt, 2*dt, ...]).
        tau_array_neg (jax.Array or None): Array of negative tau values 
                                           (e.g., [-N*dt, ..., -dt]).
        L_ops (list of jax.Array): The list of collapse operators.
        H_t_func (function): Function returning H(t, E).
        E_func (function): Function returning E(t).
    Returns:
        jnp.ndarray: The G1(t, tau) matrix, either for tau>=0 or for all tau.
    """

    # --- Case 1: Only calculate positive tau (original behavior) ---
    print("Calculating G1(t, tau >= 0)...")
    G1_positive_matrix_list = []
    for i in range(len(t_array)):
        # Call the JIT-compiled function
        G1_row = get_g1_row(
            rho_t_array[i],   # rho_t
            t_array[i],
            a_L,            # a_L
            adag_L,         # adag_L
            tau_array_pos,  # tau_array
            L_ops,          # L_ops
            H_t_func,       # H_t_func
            E_func          # E_func
        )
        G1_positive_matrix_list.append(G1_row)

    G1_positive_matrix = jnp.array(G1_positive_matrix_list)
    return G1_positive_matrix
