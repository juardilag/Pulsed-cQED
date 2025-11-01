import jax
import jax.numpy as jnp
from functools import partial
import diffrax

def von_neumann_eom_diffrax(t, rho, args):
    """
    Implements the von Neumann equation for the diffrax ODE solver.
    
    This function has the signature f(t, y, args) required by diffrax.
    
    Args:
        t (float): The current time.
        rho (jax.Array): The current density matrix (y).
        args (tuple): A tuple containing (H_t_func, E_func).
        
    Returns:
        jax.Array: The time-derivative of the density matrix (d(rho)/dt).
    """
    # Unpack the static arguments
    H_t_func, E_func = args
    
    # Get the Hamiltonian H(t)/hbar at the current time t
    H_norm = H_t_func(t, E_func)
    
    # Calculate the commutator: [H, rho] = H*rho - rho*H
    commutator = H_norm @ rho - rho @ H_norm
    
    # Calculate d(rho)/dt = -i * [H/hbar, rho]
    d_rho_dt = -1j * commutator 
    
    return d_rho_dt


#@jax.jit
def solve_dynamics(rho_initial, t_array, H_t_func, E_func, solver=diffrax.Dopri5()):
    """
    Solves the density matrix dynamics using diffrax.diffeqsolve.
    """
    
    # --- KEY FIX ---
    # Force all inputs to the solver to be 32-bit.
    # This prevents the solver's internal steps from promoting to 64-bit.
    rho_initial = rho_initial.astype(jnp.complex64)
    t_array = t_array.astype(jnp.float32)
    # --- END FIX ---

    terms = diffrax.ODETerm(von_neumann_eom_diffrax)
    t0 = t_array[0]
    t1 = t_array[-1]
    args = (H_t_func, E_func)
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
    populations_t = jnp.real(jnp.diagonal(rho_all_t, axis1=1, axis2=2))
    
    return populations_t, rho_all_t