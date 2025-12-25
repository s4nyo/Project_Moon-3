import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd

delta = 0.1
g = 9.81

def q_f(h):
    rho0 = 1.225
    Mu = 0.029
    R = 8.31
    T = 288
    return rho0 * math.exp(-(Mu * g * h) / (R * T))

def F_s(q, v):
    c = 0.3
    S = np.pi * ((10.4 / 2) ** 2)
    Fs = c * S * q * v ** 2 / 2
    return Fs

def ax_f(q, vx, alpha):
    Fs = F_s(q, vx)
    Ft_max = 2687500
    F_t_x = Ft_max * math.sin(math.radians(alpha))

    Fs_x = Fs * math.sin(math.radians(alpha))

    a = (F_t_x + Fs_x) / m - g
    return a

def ay_f(q, vy, alpha):
    Fs = F_s(q, vy)
    Ft_max = 2687500

    # Тяга по Y
    F_t_y = Ft_max * math.cos(math.radians(alpha))

    # Сопротивление
    Fs_y = Fs * math.sin(math.radians(alpha))

    a = (F_t_y + Fs_y) / m - g
    return a

def v_f(vx, vy, alpha, delta):
    q = q_f(h)
    ax = ax_f(q, vx, alpha)
    ay = ay_f(q, vy, alpha)
    vx = vx + ax * delta
    vy = vy + ay * delta
    v = math.sqrt(vx ** 2 + vy ** 2)
    return [vx, vy, v]

def alpha_f(t):
    if t < 88:
        alpha = 0
    elif t < 160:
        alpha = (85 / 72) * (t - 88)
    else:
        alpha = 85 + (5 / 76) * (t - 160)
    return alpha


# Начальные данные
h = 0
x = 0
vx = 0
vy = 0
v = 0
m = 205900
t = 0
fi = 0

# Списки для графиков
speed_data = []
time_data = []
alpha_data = []
vertical_speed_data = []
horizontal_speed_data = []

while t < 190:
    alpha = alpha_f(t)
    vf = v_f(vx, vy, fi, delta)

    vx = vf[0]
    vy = vf[1]
    v = vf[2]

    time_data.append(t)
    speed_data.append(vf[2])
    alpha_data.append(alpha)
    vertical_speed_data.append(vy)
    horizontal_speed_data.append(vx)

    t += delta

df = pd.read_excel(r"C:\Users\user\PycharmProjects\Varkt_project\Prorramms\ksp_res3.xlsx")
file_path = os.path.join(os.getcwd(), 'ksp_res3.xlsx')
time_ksp = df['time'].values
x2 = df['speed'].values
time_offset = time_ksp[0]  # первое значение времени в KSP

# Сдвигаем время KSP, чтобы оно начиналось с 0
time_ksp_shifted = time_ksp - time_offset

mask = time_ksp_shifted <= 215
time_ksp_filtered = time_ksp_shifted[mask]
speed_ksp_filtered = x2[mask]

time_math = np.array(time_data)

# График: Скорость от времени
plt.figure(2, figsize=(10, 6))
plt.plot(time_math, speed_data, color="red", label="мат. модель", linewidth=2)
plt.plot(time_ksp_filtered, speed_ksp_filtered, '-b', linewidth=1.5, label="KSP", alpha=0.8)
plt.title("Скорость ракеты", fontsize=14, pad=12)
plt.xlabel("Время (с)", fontsize=12)
plt.ylabel("Скорость (м/с)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show(block=False)

plt.show()
