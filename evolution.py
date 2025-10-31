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
        
    Returns:S
        jax.Array: The time-derivative of the density matrix (d(rho)/dt).
    """
    # Unpack the static arguments
    H_t_func, E_func = args
    
    # Get the Hamiltonian H(t)/hbar at the current time t
    # H_t_func returns complex128, so H_norm is complex128
    H_norm = H_t_func(t, E_func)

    # --- FIX 1 ---
    # Force the Hamiltonian to be complex64.
    H_norm = H_norm.astype(jnp.complex64)
    
    # --- We also must ensure rho is complex64 ---
    # The solver might pass a complex128 rho if the state gets promoted.
    rho = rho.astype(jnp.complex64)
    
    # Calculate the commutator: [H, rho] = H*rho - rho*H
    # This is now complex64 @ complex64 = complex64
    commutator = H_norm @ rho - rho @ H_norm
    
    # Calculate d(rho)/dt = -i * [H/hbar, rho]
    # This is complex64 * complex64 = complex64
    d_rho_dt = jnp.complex64(-1j) * commutator 
    
    # --- FIX 2 ---
    # As a final guarantee, cast the output to complex64.
    return d_rho_dt.astype(jnp.complex64)


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