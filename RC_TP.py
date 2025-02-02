import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


###------------- Variables-----------------####

# Surface areas
As = 95         # Surface flor(m2)
Aw = 110.7      # Surface area of walls (m2)
Afen = 8        # Surface area of windows (m2) 
Ae = 110        # External surface area of walls (m2)
taufen = 0.90   # Transmition coefficient 
alpha_e = 0.2   # Absortion coeffient external wall
alpha_i = 0.4   # Absortion coeffient internal wall

# Thermal properties
Ci = 280.0    # Thermal capacity of air (kJ/K)
Cw = 62.0e3   # Thermal capacity of walls (kJ/K)
Ke = 3.2      # External thermal resistance (kW/K) 
Kw = 25.0e-3  # Thermal resistance of walls (kW/K)
Ki = 3.2      # Internal thermal resistance (kW/K)
m_dot = 55.0e-3  # Infiltration rate (kW/K)

Tc_c = [15, 15, 15, 15, 16, 18, 19, 19, 19, 19, 16, 15,
      15, 15, 15, 16, 18, 19, 19, 19, 19, 16, 15, 15]   # setpoint temperature

Tc_k = list(map(lambda x: x + 273, Tc_c)) #  conversion from Celcius to Kelvin


# Variables for test

# -- Tex simulation 
time = np.linspace(0, 23, 24)  
temperature_max = 22 + 273
temperature_min = 5 + 273
h_peak = 14  # time of maximal temp

#  Temperture function
Tex = temperature_min + (temperature_max - temperature_min) * np.cos((time - h_peak) * np.pi / 24) ** 2

# -- Radiation simulation
amplitude = 1.2  # Max solar radiation intensity (kW/m²)
half_day = 12  # Hpeak solar radiation
width = 6  # peak width 

# Solar radiation function
E = amplitude * np.exp(-((time - half_day) ** 2) / (2 * width ** 2))

Twe = 6 + 273   # Ex. Twe 
Twi = 14 + 273  # Ex. Twi 
Ta =  14 + 273   # Ex. Twa 
X0 = [Twe,Twi,Ta]   # Initial condition


###------------- MODELO -----------------####


Ge = alpha_e * Ae / (2 * E)
Gi = taufen * alpha_i  * Afen * E

""" The model is the resolved from: """


matrix_A = [
    [-2 * ( Ke + Kw) / Cw , 2 * Ki / Cw , 0],
    [2 * Kw / Cw , -2 * (Kw + Ki) / Cw , 2 * Ki / Cw],
    [0, Ki / Ci, (Ki- m_dot) / Ci]
]

matrix_B = [
    [2 * Ke / Cw , 2 / Cw, 0],
    [0, 0, 2 / Cw],
    [m_dot / Ci, 0, 0]
]


matrix_A1 = [
    [-2 / Cw * (1 / Ke + 1 / Kw), 2 / (Cw * Ki), 0],
    [2 / (Cw * Kw), -2 / Cw * (1 / Kw + 1 / Ki), 2 / (Cw * Ki)],
    [0, 1 / (Ci * Ki), 1 / (Ci * Ki) - 1 / m_dot]
]

matrix_B1 = [
    [2 / (Cw * Ke), 2 / Cw, 0],
    [0, 0, 2 / Cw],
    [1 / (Ci * m_dot), 0, 0]
]

index = 0

def RC_model(t, X, matrix_A, matrix_B, Tex, Tc, Ge, Gi):
    global index

    if index >= len(time):
        index = len(time) - 1

    Tex_t = Tex[index]
    Ge_t = Ge[index]
    Gi_t = Gi[index]

    if X[2] > Tc_k[index]:
        P_t = 0
    else:
        P_t = 4

    D = np.array([0, 0, P_t / Ci])
    C = np.array([Tex_t, Ge_t, Gi_t])

    print(index, P_t, Tc_k[index], X[2])
    index += 1
    
    return matrix_A @ X + matrix_B @ C + D


# Time interval for evaluation
t_span = [0, 23]
t_eval = time


# Solve the system
sol = solve_ivp(RC_model, t_span, X0, t_eval=t_eval, args=(matrix_A, matrix_B, Tex, Tc_k, Ge, Gi))

# Extract the solution
time = sol.t
T_w_e, T_w_i, T_a = sol.y

# Plotting the results


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 8))  

ax[0].plot(time, T_w_e, label="Tw,e (External Wall)")
ax[0].plot(time, T_w_i, label="Tw,i (Internal Wall)")
ax[0].plot(time, T_a, label="Ta (Indoor Environment)")
ax[0].set_ylabel("Temperature (K)")
ax[0].set_title("Evolution of Temperatures Over Time")
ax[0].legend()
ax[0].grid()
ax[0].label_outer()  # Oculta etiquetas del eje X si comparte eje

ax[1].plot(time, Tc_k, label="Setpoint Temperature")
ax[1].plot(time, Tex, label="External Temperature")
ax[1].set_ylabel("Temperature (K)")
ax[1].set_title("Setpoint and External Temperature Over Time")
ax[1].legend()
ax[1].grid()
ax[1].label_outer()  # Oculta etiquetas del eje X si comparte eje

ax[2].plot(time, E, label="Solar Radiation", color='orange')
ax[2].set_xlabel("Time")
ax[2].set_ylabel("Solar Radiation (kW/m²)")
ax[2].set_title("Solar Radiation Over Time")
ax[2].legend()
ax[2].grid()

# Ajustar diseño
plt.tight_layout()
plt.show()