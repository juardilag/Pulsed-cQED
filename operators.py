import jax
import jax.numpy as jnp
from functools import partial

def boson_ops(dim : int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Creates the JAX-compatible bosonic creation, annihilation, and number operators.
    
    Args:
        dim (int): The dimension of the local Hilbert space (max_occupancy + 1).
                   
    Returns:
        tuple: A tuple of JAX arrays for annihilation (a), creation (adag),
               and number (n_op) operators.
    """
    # Create the values for the superdiagonal: sqrt(1), sqrt(2), ...
    off_diag_values = jnp.sqrt(jnp.arange(1, dim))
    a = jnp.diag(off_diag_values, k=1).astype(jnp.complex64)
    adag = a.conj().T
    n_op = adag @ a
    
    return a, adag, n_op


def qubit_ops() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Creates the JAX-compatible qubit operators (sigma_minus, sigma_plus, sigma_z).
    
    Returns:
        tuple: A tuple of JAX arrays for sigma_minus (sigma), 
               sigma_plus (sigma_dag), and sigma_z.
    """
    sigma_m = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64) # sigma
    sigma_p = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64) # sigma_dag
    sigma_e = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)
    
    return sigma_m, sigma_p, sigma_e


def static_hamiltonian(
    dim_cavity : int,
    omega_c : float, 
    omega_a : float, 
    g) -> jnp.ndarray:
    """
    Constructs the static part of the Jaynes-Cummings Hamiltonian H0 / hbar.
    Args:
        dim_cavity (int): Dimension of the cavity Hilbert space (max_occupancy + 1).
        omega_c (float): Cavity frequency.
        omega_a (float): Qubit transition frequency.
        g (float): Coupling strength between the cavity and qubit.
    Returns:
        jnp.ndarray: The static Hamiltonian matrix H0 / hbar.
        """
    a, adag, n_op = boson_ops(dim_cavity)
    sigma_m, sigma_p, sigma_e = qubit_ops()
    
    I_c = jnp.eye(dim_cavity, dtype=jnp.complex64)
    I_q = jnp.eye(2, dtype=jnp.complex64)
    
    H_c = omega_c*jnp.kron(n_op, I_q)   
    
    H_q = omega_a * jnp.kron(I_c, sigma_e)
    
    H_int = g * (jnp.kron(adag, sigma_m) + jnp.kron(a, sigma_p))
    
    H0 = H_c + H_q + H_int

    return H0


def dynamical_hamiltonian(
    dim_cavity, 
    delta_c, 
    delta_a,
    g
    ):
    """
    A function factory that creates the time-dependent Hamiltonian function H(t).
    
    This function computes the time-independent parts of the Hamiltonian once
    and returns a function that can be called with time `t` and the 
    drive envelope function E_func.
    
    The returned Hamiltonian is H(t) / hbar, assuming all parameters
    (delta_c, delta_q, g, E_func) are given in units of angular frequency.

    Args:
        dim_cavity (int): Dimension of the cavity Hilbert space (max_occupancy + 1).
        omega_c (float): Cavity frequency.
        omega_e (float): Qubit transition frequency.
        g (float): Coupling strength between the cavity and qubit.

    Returns:
        function: A function H_t(t, E_func) that computes the Hamiltonian
                  matrix at time t.
    """
    H0 = static_hamiltonian(dim_cavity, delta_c, delta_a, g)
    
    sigma_m, sigma_p, _ = qubit_ops()
    
    I_c = jnp.eye(dim_cavity)

    sigma_p_full = jnp.kron(I_c, sigma_p)
    sigma_m_full = jnp.kron(I_c, sigma_m)

    # 5. Define and return the function that computes H(t)    
    @partial(jax.jit, static_argnames=['E_func'])
    def H_t(t, E_func):
        """
        Computes the total Hamiltonian H(t)/hbar at a specific time t.

        Args:
            t (float): The current time.
            E_func (callable): A function E_func(t) that returns the 
                               complex scalar value of the drive envelope E(t).

        Returns:
            jax.Array: The total Hamiltonian matrix at time t.
        """
        # Get the scalar drive value
        E_val = E_func(t)
        H_drive = 0.5*(E_val*sigma_p_full + jnp.conj(E_val)*sigma_m_full)
        return H0 + H_drive

    return H_t