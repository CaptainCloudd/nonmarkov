import numpy as np
import matplotlib.pyplot as plt

rho_0 = np.array([[0.5,0.5],[0.5,0.5]])
def calc_rho(t, L, gamma_0):
    d = np.emath.sqrt(2*gamma_0*L - L**2)
    P_t = (np.e**(-L*t))*(np.cos((d*t)/2)+(L/d)*(np.sin((d*t)/2)))**2
    gamma_t = (1 - P_t)
    E_0 = np.array([[1,0],[0,np.sqrt(1-gamma_t)]])
    E_1 = np.array([[0,np.sqrt(gamma_t)],[0,0]])
    rho_t = (np.tensordot(E_0, np.tensordot(rho_0, E_0.conj().T, axes=([1],[1])), axes=([1],[1])) + np.tensordot(E_1, np.tensordot(rho_0, E_1.conj().T, axes=([1],[1])), axes=([1],[1])) )
    return(rho_t)

def calc_c(t, L, gamma_0):
    bell = [[0.5, 0, 0, 0.5], 
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0.5, 0, 0, 0.5]]
    bell = np.matrix(bell)
    d = np.emath.sqrt(2*gamma_0*L - L**2)
    P_t = (np.e**(-L*t))*(np.cos((d*t)/2)+(L/d)*(np.sin((d*t)/2)))
    gamma_t = (1 - P_t)
    sigma_y = np.array([[0, -1j], [1j, 0]])
    tensor_sigma_y = np.kron(sigma_y, sigma_y)
    E_0 = np.array([[1,0],[0,np.emath.sqrt(1-gamma_t)]]) 
    E_1 = np.array([[0,np.emath.sqrt(gamma_t)],[0,0]])
    M_0 = np.kron(E_0, np.identity(2))
    M_1 = np.kron(E_1, np.identity(2))
    rho2_t = np.tensordot(M_0, np.tensordot(bell, M_0.conj().T, axes=1), axes=1) + np.tensordot(M_1, np.tensordot(bell, M_1.conj().T, axes=1), axes=1)
    rho_til =  np.tensordot(tensor_sigma_y, np.tensordot(np.conj(rho2_t), tensor_sigma_y, axes=1), axes=1)
    terms_c = np.linalg.eigvals(rho_til)
    terms_c_sqrt = np.emath.sqrt(terms_c)
    maior = max(terms_c_sqrt)
    c = sum(max(0, maior - x) for x in terms_c_sqrt if x != maior)
    return c if c > 0 else 0

intervalo_tempo = np.arange(0, 50.5, 0.1)
valores_lambda_1 = np.arange(0.1, 2, 0.1)
valores_lambda_2 = np.arange(2.1, 3.1, 0.1)
intervalo_lambda = np.concatenate((valores_lambda_1, valores_lambda_2))

resultados_c_por_L = [[] for _ in intervalo_lambda]


for t in intervalo_tempo:
    for i, L in enumerate(intervalo_lambda):
        conc = calc_c(t, L, 1)
        resultados_c_por_L[i].append(conc)
for i, L in enumerate(intervalo_lambda):
    plt.plot(intervalo_tempo, resultados_c_por_L[i], label=f'L = {L}')

plt.xlabel('Tempo (s)')
plt.ylabel('Concorrência')
plt.grid(True)
plt.show()

#for t in intervalo_tempo:
    #result_rho = calc_rho(t, np.sqrt(0.7 ), 1)
    #print(result_rho)
