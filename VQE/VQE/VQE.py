import numpy as np
import cirq
import scipy.optimize

class VQE_Ising:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.n = height * width
        self.q = cirq.LineQubit.range(self.n)
        self.theta = np.zeros(5 * (2 * height * width - height - width)) # Ansatz parameters; see definition of coupling_gate()
        self.h, self.J = self.gen_random_energies(h_amp=1, J_amp=1) # External and coupling energies
        self.simulator = cirq.Simulator()
        self.measurement_reps = 100
        self.iteration = 0
        self.energy = None

    # Generate uniformly random external (h_i) and coupling (J_ij) energies
    def gen_random_energies(self, h_amp, J_amp):
        h = (np.random.rand(self.n) - 0.5) * 2 * h_amp
        J = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                xi = i % self.width
                yi = int(i / self.width)
                xj = j % self.width
                yj = int(j / self.width)

                # If qubits are exactly one step apart in the lattice
                if abs(xj - xi) + abs(yj - yi) == 1:
                    J[i, j] = (np.random.rand() - 0.5) * 2 * J_amp

        return h, J
    
    # Every two-qubit coupling gate is a general SU(4) unitary up to global phase with 2 RotX, 4 RotY
    # and 4 RotZ gates omitted to reduce parameter count
    # See Farrokh Vatan and Colin Williams, Optimal quantum circuits for general two-qubit gates (2008).
    def coupling_gate(self, q0, q1, theta):
        G = [] # Gate list

        G.append(cirq.rx(theta[0]).on(q0))
        G.append(cirq.rx(theta[1]).on(q1))
        G.append(cirq.CNOT.on(q1, q0))
        G.append(cirq.rz(theta[2]).on(q0))
        G.append(cirq.ry(theta[3]).on(q1))
        G.append(cirq.CNOT.on(q0, q1))
        G.append(cirq.ry(theta[4]).on(q1))
        G.append(cirq.CNOT.on(q1, q0))

        return G
    
    # The ansatz consists of one coupling gate for each Ising coupling in the lattice
    def U(self, theta):
        U = cirq.Circuit()  # Unitary
        G = []              # Gate list
        coupling_index = 0

        # Horizontal couplings
        for i in range(self.height):
            for j in range(self.width - 1):
                q0 = self.q[self.width * i + j    ]
                q1 = self.q[self.width * i + j + 1]
                G.append(self.coupling_gate(q0, q1, theta[5 * coupling_index:5 * (coupling_index + 1)]))
                coupling_index += 1
        # Vertical couplings
        for i in range(self.width):
            for j in range(self.height - 1):
                q0 = self.q[i + self.height * j      ]
                q1 = self.q[i + self.height * (j + 1)]
                G.append(self.coupling_gate(q0, q1, theta[5 * coupling_index:5 * (coupling_index + 1)]))
                coupling_index += 1

        #for i in range(self.n): G.append(cirq.rx(theta[i]).on(self.q[i]))

        U.append(G)
        return U

    # Compute energy for measured spin configuration
    def calc_energies(self, meas):
        s = 1 - 2 * meas # {0, 1} to {1, -1}
        energies_h = np.sum(s * self.h, axis=1)
        energies_J = np.sum((s @ self.J) * s, axis=1)

        return energies_h + energies_J

    # Objective function (expected energy) to be minimised
    def f(self, theta):
        circuit = self.U(theta)
        circuit.append([cirq.measure(self.q[j], key='{}'.format(j)) for j in range(self.n)])

        sim_result = list(self.simulator.run(circuit, repetitions=self.measurement_reps).measurements.items())
        measurements = np.hstack([r for _, r in sim_result]).astype(np.int)
        mean_energy = np.mean(self.calc_energies(measurements))
        
        print("\r({:04d}) E = {:.5}    ".format(self.iteration, mean_energy), end="")
        self.iteration += 1
        self.energy = mean_energy

        return mean_energy
    
    # Optimisation procedure
    def optimise(self):
        theta_init = (np.random.rand(len(self.theta)) * 2 - 1) * np.pi

        opt_res = scipy.optimize.minimize(self.f, theta_init, method="Powell",
                                          options={"maxiter":100, "disp":True})

        self.theta = opt_res.x
        return self.f(self.theta)

    # Brute force solution for comparison to optimisation result
    def brute_force(self):
        meas = np.zeros((0, self.n))
        for i in range(pow(2, self.n)):
            binary = bin(i)[2:].zfill(self.n)
            meas = np.vstack([meas, np.array(list(binary))])
        meas = meas.astype(int)

        min_energy = np.min(self.calc_energies(meas))

        return min_energy


vqe = VQE_Ising(height=2, width=2)

brute_force_result = vqe.brute_force()
print("Brute force minimal energy: ", brute_force_result)

optimisation_result = vqe.optimise()
print("Minimal found energy: ", optimisation_result)