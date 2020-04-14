import numpy as np
import cirq
import scipy.optimize

class QAOA_MaxCut:
    def __init__(self, n, r):
        self.n = n
        self.q = cirq.LineQubit.range(n)
        self.r = r
        self.theta = np.zeros(2 * r) # betas and gammas together in one array
        self.A, self.E = self.gen_random_edges(n, 0.7)
        self.simulator = cirq.Simulator()
        self.measurement_reps = 100

    # Generate a random graph between the n nodes
    def gen_random_edges(self, n, p):
        A = np.triu(np.random.rand(n, n), 1) # Adjacency matrix
        A[A >  1 - p] = 1
        A[A <= 1 - p] = 0
        A = A + A.T

        E = []

        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    E.append((i, j))

        return A, E
    
    def expiX(self, alpha):
        return cirq.X ** (2 * alpha / np.pi)

    def expiZZ(self, alpha):
        return cirq.ZZ ** (2 * alpha / np.pi)
    
    # QAOA circuit
    def U(self, beta, gamma):
        U = cirq.Circuit()  # Unitary
        G = []              # Gate list

        for k in range(self.r):
            for j in range(self.n):
                G.append(self.expiX(-beta[k]).on(self.q[j]))
            for e in self.E:
                G.append(self.expiZZ(-gamma[k] / 2).on(self.q[e[0]], self.q[e[1]])) # MaxCut hamiltonian up to global phase
        for j in range(self.n):
            G.append(cirq.H.on(self.q[j]))

        U.append(G)
        return U

    # Objective function: applies the circuit, measures and average calculates number of cuts
    def f(self, theta):
        beta  = theta[:self.r]
        gamma = theta[self.r:]
        circuit = self.U(beta, gamma)
        circuit.append([cirq.measure(self.q[j], key='{}'.format(j)) for j in range(self.n)])

        sim_result = list(self.simulator.run(circuit, repetitions=self.measurement_reps).measurements.items())
        measurements = np.hstack([r for _, r in sim_result]).astype(np.int)
        
        mean_cut = np.mean(self.calc_cuts(measurements))

        print(mean_cut)
        return -mean_cut
    
    # Calculate number of cuts for given partition
    def calc_cuts(self, meas):
        cuts = np.sum((meas @ self.A) * (1 - meas), axis = 1)
        return cuts
    
    # Optimise angles with respect to objective function and return optimal result
    def optimise(self):
        r = self.r
        theta_init = np.random.rand(2 * r)
        theta_init[:r] = theta_init[:r] * np.pi
        theta_init[r:] = theta_init[r:] * 2 * np.pi

        opt_res = scipy.optimize.minimize(self.f, theta_init, method="Powell",
                                          options={"maxiter":1000, "disp":True})

        self.theta = opt_res.x
        
        return -self.f(self.theta)

    # Brute force solution for comparison to optimisation result
    def brute_force(self):
        meas = np.zeros((0, self.n))
        for i in range(pow(2, self.n)):
            binary = bin(i)[2:].zfill(self.n)
            meas = np.vstack([meas, np.array(list(binary))])
        meas = meas.astype(int)

        max_cuts = np.max(self.calc_cuts(meas))

        return max_cuts

qaoa = QAOA_MaxCut(n=2, r=2)

brute_force_result = qaoa.brute_force()
print("Brute force maximum cut: ", brute_force_result)

optimisation_result = qaoa.optimise()
print("Maximum cut found: ", optimisation_result)