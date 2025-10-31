import jax
import jax.numpy as jnp
from operators import boson_ops, qubit_ops

def output_observables(
    cavity_dim: int,
    projector_list: list[jnp.ndarray],
    rho_t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepares commonly used observables for the cavity-qubit system.
    
    Returns:
        tuple: A tuple containing the photon number operator for the cavity (N_cav)
               and the excited-state projector for the qubit (P_e).
    """
    # 1. Create local operators

    a, adag, n_op = boson_ops(cavity_dim)
    sigma_m, sigma_p, sigma_z = qubit_ops()

    # 2. Get identity operators for each subspace
    I_c = jnp.eye(cavity_dim, dtype=jnp.complex64)
    I_q = jnp.eye(2, dtype=jnp.complex64)

    # 3. Create the full operators in the tensor product space
    #    We must match the kron order from your Hamiltonian function:
    #    (cavity, qubit)

    # Photon number operator: n_op_cav * I_q
    N_cav = jnp.kron(n_op, I_q)

    # Atomic excited-state projector: I_c * |e><e|
    # |e><e| = (sigma_z + I) / 2
    sigma_e = (sigma_z + I_q) / 2.0
    P_e = jnp.kron(I_c, sigma_e)

    atom_pop = jnp.trace(P_e @ rho_t, axis1=1, axis2=2)
    light_pop = jnp.trace(N_cav @ rho_t, axis1=1, axis2=2)

    projector_observables = []

    for projector in projector_list:
        pop = jnp.trace(projector @ rho_t, axis1=1, axis2=2)
        projector_observables.append(pop)

    return light_pop, atom_pop, projector_observables