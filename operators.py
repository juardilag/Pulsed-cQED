import jax.numpy as jnp

def boson_ops(dim : int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Creates bosonic creation, annihilation, and number operators.
    
    Args:
        dim (int): The dimension of the local Hilbert space (max_occupancy + 1).
                   
    Returns:
        tuple: A tuple of JAX arrays for annihilation (a), creation (adag),
               and number (n_op) operators.
    """
    # Create the values for the superdiagonal: sqrt(1), sqrt(2), ...
    off_diag_values = jnp.sqrt(jnp.arange(1, dim))
    
    # Construct the annihilation operator directly
    a = jnp.diag(off_diag_values, k=1).astype(jnp.complex64)
    
    # Creation operator is the conjugate transpose
    adag = a.conj().T
    
    # Number operator
    n_op = adag @ a
    
    return a, adag, n_op


def qubit_ops() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Creates the qubit operators (sigma_minus, sigma_plus, sigma_ee).
    
    Returns:
        tuple: A tuple of JAX arrays for sigma_minus (sigma), 
               sigma_plus (sigma_dag), and sigma_ee.
    """
    sigma_ee = jnp.array([[0, 0], [0, 1]], dtype=jnp.complex64)
    sigma_m = jnp.array([[0, 0], [1, 0]], dtype=jnp.complex64) # sigma
    sigma_p = jnp.array([[0, 1], [0, 0]], dtype=jnp.complex64) # sigma_dag
    
    return sigma_m, sigma_p, sigma_ee