import numpy as np

def input(n, index):
    N = 2**n
    vec = np.zeros(N)
    vec[index] = 1
    return vec

def hadamard(n):
    H = np.array([[1, 1], [1, -1]])*(1/np.sqrt(2))
    M = np.array([[1, 1], [1, -1]])*(1/np.sqrt(2))
    for _ in range (n-1):
        H = np.kron(H, M)
    return H

def oracle_matrix(n, f):
    N = 2**n
    matrix = np.identity(N)
    for i in range(N):
        if f(i) == 1:
            matrix[i, i] = -1
    return matrix

def diffusion_operator(n):
    N = 2**n

    H_n = hadamard(n)
    zero_state = input(n, 0)
    psi = np.matmul(H_n, zero_state)

    diffusion = 2 * np.outer(psi, psi) - np.identity(N)
    return diffusion

def grover_algorithm(n, f):
    N = 2**n
    # Número de soluciones marcadas por el oráculo (Siempre yes uno en este caso)
    marked_count = sum(f(i) for i in range(N))

    if marked_count == 0:
        print("No hay soluciones marcadas por el oráculo.")
        return None

    iterations = int(np.pi/4 * np.sqrt(N/marked_count))
    print(f"Realizando {iterations} iteraciones de Grover.")

    # Aplicar Hadamard
    H_n = hadamard(n)
    state = np.matmul(H_n, input(n, 0))

    # Crear matrices para el oráculo y el operador de difusión
    U_f = oracle_matrix(n, f)
    U_s = diffusion_operator(n)

    # Aplicar iteraciones de Grover
    for i in range(iterations):
        state = np.matmul(U_f, state)
        state = np.matmul(U_s, state)

    probabilities = np.abs(state)**2
    most_probable = np.argmax(probabilities)

    return most_probable, probabilities

if __name__ == "__main__":
    n = 5

    def example_oracle(x):
        return 1 if x == 3 else 0

    result, probabilities = grover_algorithm(n, example_oracle)

    print(f"Resultado más probable: |{bin(result)[2:].zfill(n)}⟩")
    print(f"Con probabilidad: {probabilities[result]:.4f}")
