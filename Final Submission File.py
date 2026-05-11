import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from numpy import lib

# universal constants
k_B = 1.0
elementary_charge = 1.0
delta_T = 0.01
plancks_constant = 2 * np.pi

# truncation parameter
N = 0

# initialisation of constants
E_0 = 1.0
mu = 1.0
omega = 0.2*mu
Delta_SC = 0.1
eta = 0.4
v_param = 0.4
W_0 = 2 * np.sqrt(mu) * eta  # W values chosen to reflect the ratio used in the semiconductor example
W_1 = v_param * W_0

# initialisation of energy spectrum
n = np.arange(-N, N+1, 1)
E_n = E_0 + n * omega


# q defined for normal metals
def q_n(E, Mu):
    # q^+_n
    return lib.scimath.sqrt(Mu + E)


def qp_n(E, Mu):
    # q^-_n
    return lib.scimath.sqrt(Mu - E)


# k defined for superconductor
def k_n(E, Mu, Delta):
    # k^+_n
    return lib.scimath.sqrt(Mu + np.sign(E)*(lib.scimath.sqrt(E**2 - Delta**2)))


def kp_n(E, Mu, Delta):
    # k^-_n
    return lib.scimath.sqrt(Mu + np.sign(E)*(-lib.scimath.sqrt(E**2 - Delta**2)))


# electron and hole mixing terms
@np.vectorize  # allows for comparison of individual elements to remove E = 0 errors.
def u0_n(E, Delta):
    if E == 0:
        return np.sqrt(1/2)
    else:
        return lib.scimath.sqrt(0.5 * (1 + lib.scimath.sqrt((E**2 - Delta**2)/E**2)))


@np.vectorize
def v0_n(E, Delta):
    if E == 0:
        return np.sqrt(1/2)
    else:
        return lib.scimath.sqrt(0.5 * (1 - lib.scimath.sqrt((E**2 - Delta**2)/E**2)))


def minus_fermi_prime(epsilon, T0):
    z = np.clip(epsilon / (2.0 * k_B * T0), -50, 50)
    return 1.0 / (4.0 * k_B * T0 * np.cosh(z)**2)


def M_nn(n, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_0):

    index = n + N

    qm = q_minus[index]
    qp = q_plus[index]
    km = k_minus[index]
    kp = k_plus[index]
    u = u0[index]
    v = v0[index]
    #print(qm)
    #print(qp)
    #print(km)
    #print(kp)
    return np.array([[1/lib.scimath.sqrt(qm), 0, -v/lib.scimath.sqrt(kp), -u/lib.scimath.sqrt(km)],
                            [0, 1/lib.scimath.sqrt(qp), -u/lib.scimath.sqrt(kp), -v/lib.scimath.sqrt(km)],
                            [-(W_0/lib.scimath.sqrt(qm)) - (1j * lib.scimath.sqrt(qm)), 0, 1j * lib.scimath.sqrt(kp)*v, -1j * lib.scimath.sqrt(km)*u],
                            [0, 1j * lib.scimath.sqrt(qp), (1j * lib.scimath.sqrt(kp)*u) - (W_0 * u/lib.scimath.sqrt(kp)), (-1j * lib.scimath.sqrt(km)*v)
                             - W_0 * v/lib.scimath.sqrt(km)]], dtype=complex)


def M_n_n_minus_1(n, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_1):

    j = (n - 1) + N

    qm = q_minus[j]
    km = k_minus[j]
    kp = k_plus[j]
    u = u0[j]
    v = v0[j]
    return np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-W_1/lib.scimath.sqrt(qm), 0, 0, 0],
                            [0, 0, -W_1*u/lib.scimath.sqrt(kp), -W_1*v/lib.scimath.sqrt(km)]], dtype=complex)


def M_n_n_plus_1(n, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_1):

    j = (n + 1) + N

    qm = q_minus[j]
    km = k_minus[j]
    kp = k_plus[j]
    u = u0[j]
    v = v0[j]
    return np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-W_1/lib.scimath.sqrt(qm), 0, 0, 0],
                            [0, 0, -W_1*u/lib.scimath.sqrt(kp), -W_1*v/lib.scimath.sqrt(km)]], dtype=complex)


def construct_matrix(n, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_0, W_1):

    dim = 4*(2*N + 1)
    M = np.zeros((dim, dim), dtype=complex)

    def block_index(k):
        return 4*(k + N)

    for k in n:
        i = block_index(k)

        # diagonal block
        M[i:i+4, i:i+4] = M_nn(k, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_0)

        # lower diagonal block
        if k > -N:
            j = block_index(k - 1)
            M[i:i+4, j:j+4] = M_n_n_minus_1(k, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_1)

        if k < N:
            j = block_index(k + 1)
            M[i:i+4, j:j+4] = M_n_n_plus_1(k, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_1)
    return M


def construct_input_vector(N, inputs=[]):

    dim = 4*(2*N + 1)
    B = np.zeros(dim, dtype=complex)
    amp_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

    for n_in, amp, val in inputs:
        i = 4*(n_in + N) + amp_dict[amp]
        B[i] = val

    return B


E_start_value = -0.59
E_stop_value = 0.59
num_intervals = 500

# defining an energy input spectrum
E_values = np.linspace(E_start_value, E_stop_value, num_intervals)

# initialising data storage
a_coeffs = []
little_b_coefficients = []
b_coeffs = []
c_coeffs = []
d_coeffs = []
total_sum = []

for E_val in E_values:

    epsilon = E_val + n * omega
    E_n = epsilon

    mask = E_n + mu >= 0

    q_minus = qp_n(E_n, mu)
    q_plus = q_n(E_n, mu)
    q_plus_n = q_n(E_n[N], mu)

    k_minus = kp_n(E_n, mu, Delta_SC)
    k_plus = k_n(E_n, mu, Delta_SC)
    u0 = u0_n(E_n, Delta_SC)
    v0 = v0_n(E_n, Delta_SC)

    M = construct_matrix(n, N, q_minus, q_plus, k_minus, k_plus, u0, v0, W_0, W_1)

    # removing negative channels from the matrix
    for j in range(2*N + 1):
        if not mask[j]:
            idx = 4*j
            M[idx:idx+4, :] = 0
            M[:, idx:idx+4] = 0
            M[idx:idx+4, idx:idx+4] = np.eye(4)

    B = construct_input_vector(N, inputs=[(0, 'b', -1 / lib.scimath.sqrt(q_plus_n)),
                                          (0, 'd', 1j * lib.scimath.sqrt(q_plus_n))])

    # removing negative channels from the vector
    for j in range(2*N + 1):
        if not mask[j]:
            idx = 4*j
            B[idx:idx+4] = 0
    #print(v0)
    #print(M)

    Phi = solve(M, B)

    A_sum = 0
    B_sum = 0
    C_sum = 0
    D_sum = 0

    for j in range(2 * N + 1):

        # implementing the skip
        if not mask[j]:
            continue

        a_n = Phi[4 * j]
        b_n = Phi[4 * j + 1]
        c_n = Phi[4 * j + 2]
        d_n = Phi[4 * j + 3]

        if np.abs(epsilon[j]) < Delta_SC:
            c_n = 0
            d_n = 0

        u = u0[j]
        v = v0[j]

        uv_factor = u**2 - v**2

        A_sum += np.abs(a_n) ** 2
        B_sum += np.abs(b_n) ** 2
        little_b_coefficients.append(b_n)
        C_sum += uv_factor * np.abs(c_n) ** 2
        D_sum += uv_factor * np.abs(d_n) ** 2

    total_current = A_sum + B_sum + C_sum + D_sum

    #if total_current > 1e-12:
        #A_sum /= total_current
        #B_sum /= total_current
        #C_sum /= total_current
        #D_sum /= total_current

    a_coeffs.append(A_sum)
    b_coeffs.append(B_sum)
    c_coeffs.append(C_sum)
    d_coeffs.append(D_sum)
    total_sum.append(total_current)

    print("Cycle", ((E_val - E_start_value)/(E_stop_value - E_start_value))*100, '%:', A_sum + B_sum + C_sum + D_sum)


A_array = np.asarray(a_coeffs, dtype=complex)
B_array = np.asarray(b_coeffs, dtype=complex)
C_array = np.asarray(c_coeffs, dtype=complex)
D_array = np.asarray(d_coeffs, dtype=complex)

current_kernel = C_array + D_array + (-1 * A_array) + (-1 * B_array)
#print(current_kernel)

def induced_current(T0):
    thermal_weight = E_values * minus_fermi_prime(E_values, T0)

    integrand = current_kernel * thermal_weight

    integral = np.trapezoid(integrand, E_values)

    prefactor = elementary_charge * delta_T / (2 * T0 * plancks_constant)

    return prefactor * integral


T0_values = np.linspace(0.005, 0.2, num_intervals)

induced_current_values = induced_current(T0_values)

if N == 0:
    output_filename = "static_barrier_SC_current_results.npz"
else:
    output_filename = "driven_barrier_SC_current_results.npz"


np.savez(
    output_filename,
    I_values=induced_current_values
)

plt.figure(figsize=(8, 5))
x=E_values/mu
plt.plot(E_values, a_coeffs, label='Andreev ($|a|^2$)', lw=2, color='blue')
plt.plot(E_values, b_coeffs, label='Refl. Electron ($|b|^2$)', lw=2, color='orange')
plt.plot(E_values, c_coeffs, label='Trans. Electron ($\\kappa |c|^2$)', linestyle='-', color='green')
plt.plot(E_values, d_coeffs, label='Trans. Hole ($\\kappa |d|^2$)', linestyle='-', color='red')
plt.plot(E_values, total_sum, label='Total Probability (A + B + C + D)', lw=2, linestyle='--', color='purple')
#print(little_b_coefficients)
plt.axvspan(E_start_value/mu, -mu, alpha=0.25, color='gray', label='Closed Channel Region')
plt.axvline(Delta_SC, color='k', linestyle=':', label=r'Gap Edge ($\pm \Delta_{SC}$)')
plt.axvline(-Delta_SC, color='k', linestyle=':')
plt.xlim(E_start_value, (x[-1]+0.01))
#plt.xlim(0, 0.4)
#plt.ylim(0.0, 0.003)
plt.xlabel('Incident Energy $E_0/\\mu$')
plt.ylabel('Probability')
plt.title('Extended Axes for the Barrier-less NS Junction')
plt.tight_layout()
plt.legend(loc='center right')
#plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(T0_values, induced_current_values, lw=2, label='Induced Current', color='blue')
plt.xlabel(r"Initial temperature $T_0$")
plt.ylabel(r"Induced current $\bar{I}^{e}_{\delta T}$")
plt.title(r"Temperature Gradient Induced Current Against $T_0$ Driven Case")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()

static_data = np.load("static_barrier_SC_current_results.npz")
driven_data = np.load("driven_barrier_SC_current_results.npz")

I_static = static_data["I_values"]
I_driven = driven_data["I_values"]

delta_I = I_driven - I_static

plt.figure(figsize=(8, 5))
plt.plot(T0_values, delta_I, lw=2, label=r"$I_{\mathrm{driven}} - I_{\mathrm{static}}$", color='red')
plt.axhline(0, color="black", linestyle="-", lw=1)
plt.xlabel(r"Initial temperature $T_0$")
plt.ylabel(r"Change in induced current $\Delta \bar{I}^{e}_{\delta T}$")
plt.title(r"Change in Temperature Gradient Induced Current Static vs. Driven")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.show()
