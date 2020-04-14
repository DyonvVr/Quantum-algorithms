import numpy as np
import cirq

class Grover:
    def __init__(self, n):
        self.n = n                                      # Number of qubits
        self.index = 1#np.random.randint(pow(2, n))       # Choose random number in [0, 2^n) which is regarded as the "solution"
        self.qubits = cirq.LineQubit.range(2 * self.n)  # All qubits
        self.q = self.qubits[0:self.n]                  # Grover work qubits
        self.a = self.qubits[self.n:self.n + 1]         # Grover ancilla qubit
        self.t = self.qubits[self.n + 1:2 * self.n]     # Toffoli ancilla qubits
        self.simulator = cirq.Simulator()
        self.measurement_reps = 1000

    def multi_toffoli(self, qins, qout, controls=None):
        # Array containing whether i-th qubit should have a 0-control (i.e. added X gates) or 1-control
        if controls is None: controls = np.ones(self.n) # If none specified, use default 1-controls

        G = []
        Xgates = []
        Cgates = []

        for i in range(len(qins)):
            if controls[i] == 0: Xgates.append(cirq.X.on(qins[i]))

        if len(qins) > 1: Cgates.append(cirq.TOFFOLI.on(qins[0], qins[1], self.t[0]))
        for i in range(2, len(qins)):
            Cgates.append(cirq.TOFFOLI.on(qins[i], self.t[i - 2], self.t[i - 1]))
        #for i in range(2, len(qins) - 1):
        #    Cgates.append(cirq.TOFFOLI.on(qins[i], self.t[i - 2], self.t[i - 1]))

        G.append(Xgates)
        G.append(Cgates)
        if len(qins) > 1:
            G.append(cirq.CNOT.on(self.t[len(qins) - 2], qout))
            #G.append(cirq.TOFFOLI.on(qins[len(qins) - 1], self.t[len(qins) - 3], qout))
        else:
            G.append(cirq.CNOT.on(qins[0], qout))
        G.append(cirq.inverse(Cgates)) # Uncompute t-registers
        G.append(Xgates)

        return G

    def oracle(self):
        index_binary = np.array(list(bin(self.index)[2:].zfill(self.n))).astype(int)
        
        return self.multi_toffoli(self.q, self.a[0], controls=index_binary)
    
    def diffusion(self):
        G = [] # Gate list

        for i in range(self.n):
            G.append(cirq.H.on(self.q[i]))

        # 0-controlled Z on all work qubits
        G.append(cirq.X.on(self.q[-1]))
        G.append(cirq.H.on(self.q[-1]))
        G.append(self.multi_toffoli(self.q[:-1], self.q[-1], controls=np.zeros(self.n)))
        G.append(cirq.H.on(self.q[-1]))
        G.append(cirq.X.on(self.q[-1]))

        for i in range(self.n):
            G.append(cirq.H.on(self.q[i]))

        return G
    
    def search(self):
        circuit = cirq.Circuit()
        
        # Grover ancilla starts in |1> state
        circuit.append(cirq.X.on(self.a[0]))

        for i in range(self.n):
            circuit.append(cirq.H.on(self.q[i]))
        circuit.append(cirq.H.on(self.a[0]))

        grover_reps = int(np.round(np.pi / 4 * np.sqrt(pow(2, self.n))))

        for i in range(grover_reps):
            circuit.append(self.oracle())
            circuit.append(self.diffusion())
            
        circuit.append([cirq.measure(self.q[j], key='{}'.format(j)) for j in range(self.n)])

        sim_result = list(self.simulator.run(circuit, repetitions=self.measurement_reps).measurements.items())
        measurements = np.hstack([r for _, r in sim_result]).astype(np.int)
        results = measurements.dot(1 << np.arange(measurements.shape[-1] - 1, -1, -1)) # Convert binary to decimal

        print(measurements)
        print(results)

        accuracy = np.sum(results == self.index) / self.measurement_reps

        return accuracy

grover = Grover(3)
print(grover.index)
print(cirq.Circuit(grover.oracle()))
print(cirq.Circuit(grover.diffusion()))
print(grover.search())