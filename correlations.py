from evolution import solve_dynamics
import jax
import jax.numpy as jnp
from functools import partial

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
    # 0. Definitions
    n_op = adag_L @ a_L
    
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
    
    # Denominator (Intensities)
    n_t = jnp.real(calculate_trace(rho_t, n_op))
    n_t_plus_tau = jnp.real(calculate_trace_vmap(rho_tau, n_op))
    epsilon_sq = 1e-3 
    denominator = jnp.sqrt(n_t*n_t_plus_tau + epsilon_sq)

    # --- REGULARIZATION ---
    # epsilon corresponds to the "noise floor" intensity.
    g1_regularized = numerator / (denominator )
    return numerator, denominator, g1_regularized

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

    numerator_list = []
    denominator_list = []
    g2_2d_list = []

    for i in range(len(t_array)):
        # Call the JIT-compiled function
        numerator, denominator, g1_regularized = get_g1_row(
            rho_t_array[i],   # rho_t
            t_array[i],
            a_L,            # a_L
            adag_L,         # adag_L
            tau_array_pos,  # tau_array
            L_ops,          # L_ops
            H_t_func,       # H_t_func
            E_func          # E_func
        )
        numerator_list.append(numerator)
        denominator_list.append(denominator)
        g2_2d_list.append(g1_regularized)

    # Convert to arrays
    Num_matrix = jnp.array(numerator_list)   # Shape (N_t, N_tau)
    Denom_matrix = jnp.array(denominator_list) # Shape (N_t, N_tau)
    g1_2d_matrix = jnp.array(g2_2d_list)
    return Num_matrix, Denom_matrix, g1_2d_matrix


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
    
    # 0. Definitions
    n_op = adag_L @ a_L
    
    # 1. Init Conditions
    chi_0 = a_L @ rho_t @ adag_L
    rho_0 = rho_t
    stacked_0 = jnp.stack([chi_0, rho_0])

    # 2. Dynamics
    H_tau_func = lambda tau, E_f: H_t_func(t + tau, E_f)
    E_tau_func = lambda tau: E_func(t + tau)
    
    solve_dynamics_vmap = jax.vmap(solve_dynamics, in_axes=(0, None, None, None, None))
    stacked_all_tau = solve_dynamics_vmap(stacked_0, tau_array, H_tau_func, E_tau_func, L_ops)
    
    chi_tau, rho_tau = stacked_all_tau[0], stacked_all_tau[1]

    # 3. Observables
    calculate_trace_vmap = jax.vmap(calculate_trace, in_axes=(0, None))
    
    # Numerator (Correlation)
    numerator = jnp.real(calculate_trace_vmap(chi_tau, n_op))
    
    # Denominator (Intensities)
    n_t = jnp.real(calculate_trace(rho_t, n_op))
    n_t_plus_tau = jnp.real(calculate_trace_vmap(rho_tau, n_op))
    denominator = n_t*n_t_plus_tau

    # --- REGULARIZATION ---
    # epsilon corresponds to the "noise floor" intensity.
    epsilon_sq = 1e-3 
    g2_regularized = numerator / (denominator + epsilon_sq)
    
    return numerator, denominator, g2_regularized

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
    numerator_list = []
    denominator_list = []
    g2_2d_list = []

    for i in range(len(t_array)):
        # Unpack the 3 values
        raw_num, raw_denom, g2_row_plot = get_g2_row(
            rho_t_array[i], t_array[i], a_L, adag_L, 
            tau_array_pos, L_ops, H_t_func, E_func
        )
        
        numerator_list.append(raw_num)      # Save Numerator
        denominator_list.append(raw_denom)  # Save Denominator
        g2_2d_list.append(g2_row_plot)      # Save Plot version

    # Convert to arrays
    Num_matrix = jnp.array(numerator_list)   # Shape (N_t, N_tau)
    Denom_matrix = jnp.array(denominator_list) # Shape (N_t, N_tau)
    g2_2d_matrix = jnp.array(g2_2d_list)
    return Num_matrix, Denom_matrix, g2_2d_matrix