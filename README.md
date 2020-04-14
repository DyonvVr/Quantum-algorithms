# Quantum algorithms
Implementations of the quantum approximate optimisation algorithm (QAOA), a variational quantum eigensolver (VQE) and Grover's algorithm in Cirq.

The survey _Quantum algorithm implementations for beginners_ by P.J. Coles et al. (2018) was used for reference with regards to QAOA and Grover's algorithm.

## QAOA
The QAOA algorithm runs on the MaxCut problem, aiming to divide the vertices of a random graph into two disjoint subsets such that the cut between the two sets is maximal. This implementation was inspired by the [tutorial](https://github.com/quantumlib/Cirq/blob/master/examples/qaoa.py) on MaxCut QAOA in the Cirq repository.
