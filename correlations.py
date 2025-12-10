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
    Calculates a single row of the NORMALIZED g1(t, tau) matrix.
    
    It evolves both the regression matrix 'sigma' (for the numerator)
    and the density matrix 'rho' (for the denominator n(t+tau)) simultaneously.
    """
    
    # 1. Setup Initial Conditions
    sigma_0 = a_L @ rho_t
    rho_0 = rho_t
    stacked_0 = jnp.stack([sigma_0, rho_0])

    # 2. Time functions
    H_tau_func = lambda tau, E_f: H_t_func(t + tau, E_f)
    E_tau_func = lambda tau: E_func(t + tau)
    
    # 3. Solve Parallel Dynamics
    solve_dynamics_vmap = jax.vmap(solve_dynamics, in_axes=(0, None, None, None, None))
    stacked_all_tau = solve_dynamics_vmap(
        stacked_0, 
        tau_array, 
        H_tau_func, 
        E_tau_func, 
        L_ops
    )
    
    sigma_tau = stacked_all_tau[0]
    rho_tau   = stacked_all_tau[1]

    # 4. Calculate Observables
    calculate_trace_vmap = jax.vmap(calculate_trace, in_axes=(0, None))
    numerator = calculate_trace_vmap(sigma_tau, adag_L)
    
    n_t = calculate_trace(rho_t, adag_L @ a_L)
    n_t_plus_tau = calculate_trace_vmap(rho_tau, adag_L @ a_L)
    
    # 5. Normalize with Robust Protection
    pop_product = jnp.real(n_t * n_t_plus_tau)
    
    # A slightly higher threshold to catch the "stripes" earlier
    threshold = 1e-6
    
    # Soft denominator to prevent explosion
    denom = jnp.sqrt(pop_product + 1e-18)
    
    # Raw calculation
    g1_raw = numerator / denom
    
    # FILTER 1: If population is too low, set correlation to 0 (vacuum has no coherence)
    g1_masked = jnp.where(pop_product < threshold, 0.0, g1_raw)
    
    # FILTER 2: CLIPPING (Crucial Step)
    # Physically, modulus |g1| cannot exceed 1.0. 
    # We clip the absolute value magnitude while preserving phase (if complex)
    # But since you are plotting jnp.abs(g1), we can just return the array
    # and let the user clip the magnitude later, OR clip strictly here.
    
    # Here is a strict magnitude clip:
    mag = jnp.abs(g1_masked)
    phase = jnp.angle(g1_masked)
    
    # If magnitude > 1.0, force it to 1.0
    clean_mag = jnp.minimum(mag, 1.0)
    
    # Reconstruct (optional, if you need phase later)
    g1_clean = clean_mag * jnp.exp(1j * phase)
    
    return g1_clean

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


@partial(jax.jit, static_argnames=("H_t_func", "E_func"))
def get_g2_row(
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
    Calculates a single row of the unnormalized G2(t, tau) matrix for a given t.
    
    Corresponds to: <a^dag(t) a^dag(t+tau) a(t+tau) a(t)>
    
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
        jax.Array: The row of G2(t, tau) [unnormalized] for the given t.
    """
    
    # 0. Define the number operator for the final trace
    n_op = adag_L @ a_L

    # 1. QRT initial condition for G2: chi(t, 0) = a * rho(t) * a_dag
    # This represents the state immediately after a photon detection at time t.
    chi_0 = a_L @ rho_t @ adag_L
    
    # The QRT evolution from t to t+tau uses the same shifted Hamiltonian 
    # logic as the G1 implementation.
    
    # H_tau_func(tau, E) will call H_t_func(t + tau, E)
    H_tau_func = lambda tau, E_f: H_t_func(t + tau, E_f)
    # E_tau_func(tau) will call E_func(t + tau)
    E_tau_func = lambda tau: E_func(t + tau)
    
    # 2. Solve QRT evolution: d(chi)/d(tau) = L(chi)
    # The dynamics are identical to G1, just acting on a different initial state.
    chi_all_tau = solve_dynamics(
        chi_0, 
        tau_array, 
        H_tau_func,  
        E_tau_func,  
        L_ops
    )
    
    # 3. Calculate G2(t, tau) = Tr(n * chi(t, tau))
    # We trace against the number operator n = a_dag * a
    G2_row = jax.vmap(calculate_trace, in_axes=(0, None))(chi_all_tau, n_op)
    
    return G2_row

def g2_matrix(
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
    Calculates the full unnormalized G2(t, tau) matrix.
    
    To get the normalized g2(t, tau), you must divide this result by:
    <n(t)> * <n(t+tau)>
    
    Args:
        rho_t_array (jax.Array): Array of density matrices at times t_array.
        t_array (jax.Array): Array of absolute times t.
        a_L (jax.Array): The annihilation operator.
        adag_L (jax.Array): The creation operator.
        tau_array_pos (jax.Array): Array of non-negative tau values.
        L_ops (list of jax.Array): The list of collapse operators.
        H_t_func (function): Function returning H(t, E).
        E_func (function): Function returning E(t).
    Returns:
        jnp.ndarray: The unnormalized G2 matrix.
    """

    print("Calculating G2(t, tau >= 0)...")
    G2_matrix_list = []
    
    # Depending on memory, you might want to jax.lax.scan this, 
    # but a Python loop is safer for debugging/printing progress.
    for i in range(len(t_array)):
        G2_row = get_g2_row(
            rho_t_array[i],   # rho_t
            t_array[i],
            a_L,            
            adag_L,         
            tau_array_pos,  
            L_ops,          
            H_t_func,       
            E_func          
        )
        G2_matrix_list.append(G2_row)

    G2_matrix = jnp.array(G2_matrix_list)
    return G2_matrix