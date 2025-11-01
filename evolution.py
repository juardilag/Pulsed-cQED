import jax
import jax.numpy as jnp
from functools import partial
import diffrax

def lindblad_me_diffrax(t, rho, args):
    """
    Implements the Lindblad Master Equation for the diffrax ODE solver.
    
    This function has the signature f(t, y, args) required by diffrax.
    
    Args:
        t (float): The current time.
        rho (jax.Array): The current density matrix (y).
        args (tuple): A tuple containing (H_t_func, E_func, L_ops).
        
    Returns:
        jax.Array: The time-derivative of the density matrix (d(rho)/dt).
    """
    # Unpack the static arguments
    H_t_func, E_func, L_ops = args
    
    # Get the Hamiltonian H(t)/hbar at the current time t
    H_norm = H_t_func(t, E_func)
    
    # --- 1. Unitary Evolution: -i[H, rho] ---
    # Calculate the commutator: [H, rho] = H*rho - rho*H
    commutator = H_norm @ rho - rho @ H_norm
    d_rho_dt_unitary = -1j * commutator
    
    # --- 2. Dissipative Evolution (Lindblad terms) ---
    d_rho_dt_dissipative = jnp.zeros_like(rho)
    
    for L in L_ops:
        L_dag = L.T.conj()
        L_dag_L = L_dag @ L
        
        # L*rho*L_dag
        term1 = L @ rho @ L_dag
        
        # -0.5 * (L_dag*L*rho + rho*L_dag*L)
        anticommutator = L_dag_L @ rho + rho @ L_dag_L
        term2 = -0.5 * anticommutator
        
        d_rho_dt_dissipative += (term1 + term2)
        
    # Total derivative
    d_rho_dt = d_rho_dt_unitary + d_rho_dt_dissipative
    
    return d_rho_dt


#@jax.jit
def solve_dynamics(rho_initial, t_array, H_t_func, E_func, L_ops, solver=diffrax.Dopri5()):
    """
    Solves the density matrix dynamics using the Lindblad Master Equation.
    
    Args:
        rho_initial (jax.Array): Initial density matrix.
        t_array (jax.Array): Array of times to save the solution at.
        H_t_func (callable): Function H(t, E) that returns the Hamiltonian.
        E_func (callable): Function E(t) for the drive (can be None).
        L_ops (list): A list of JAX arrays, [L_1, L_2, ...], for the collapse operators.
        solver (diffrax.AbstractSolver): The ODE solver.
    """
    
    # Force all inputs to the solver to be 32-bit (as in original)
    rho_initial = rho_initial.astype(jnp.complex64)
    t_array = t_array.astype(jnp.float32)

    # Use the new Lindblad EOM function
    terms = diffrax.ODETerm(lindblad_me_diffrax)
    t0 = t_array[0]
    t1 = t_array[-1]
    
    # --- KEY CHANGE ---
    # Pass H_t_func, E_func, AND L_ops to the solver
    args = (H_t_func, E_func, L_ops)
    # --- END CHANGE ---
    
    saveat = diffrax.SaveAt(ts=t_array)
    dt0 = None 
    stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
    
    sol = diffrax.diffeqsolve(terms,
                            solver,
                            t0,
                            t1,
                            dt0,
                            rho_initial,
                            saveat=saveat,
                            args=args,
                            stepsize_controller=stepsize_controller, 
                            max_steps=16**4)
                        
    rho_all_t = sol.ys
    
    # Return the full density matrix history
    return rho_all_t