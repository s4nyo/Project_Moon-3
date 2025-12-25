import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import constants
import pandas as pd
import os

# Константы
h = 0
T = 288
Cf = 0.3
ro = 1.293
S = np.pi * ((2.1 / 2) ** 2)
g = constants.g
F_thrust = [2649660, 860000]


def angle_rad(t):
    if t <= 70:
        deg = 0
    elif t <= 140:
        deg = (45 / 70) * (t - 70)
    else:
        deg = 45 + (45 / 76) * (t - 140)
    return np.radians(deg)


def rocket_dynamics(t, y):
    vx, vy = y
    v_total = np.sqrt(vx ** 2 + vy ** 2)
    h = 0
    # Определение параметров ступени
    if t <= 88:
        R = 8.31
        M, m = 205900, 109900
        thrust = F_thrust[0]
        dt_stage = 88
        h += vy * 0.01
        ro = 1.293
    else:
        R = 8.31
        M, m = 67600, 27700
        thrust = F_thrust[1]
        dt_stage = 146
        h += vy * 0.01
        t = t - 88  # локальное время ступени
        ro = 0.8

    k = m / dt_stage
    current_mass = M - k * t

    # Сила сопротивления (упрощенно)
    f_drag = 0.5 * ro * S * (v_total ** 2) * Cf

    # Проекции ускорения
    theta = angle_rad(t)

    # Ускорение тяги
    ax_thrust = (thrust * np.sin(theta)) / current_mass
    ay_thrust = (thrust * np.cos(theta)) / current_mass

    # Ускорение сопротивления (действует против вектора скорости)
    if v_total > 0:
        ax_drag = -(f_drag * (vx / v_total)) / current_mass
        ay_drag = -(f_drag * (vy / v_total)) / current_mass
    else:
        ax_drag = ay_drag = 0

    dvx_dt = ax_thrust + ax_drag
    dvy_dt = ay_thrust + ay_drag - g

    return [dvx_dt, dvy_dt]

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

# Построение графика сравнения
plt.figure(figsize=(14, 8))
# Решение
t_span = (0, 200)
t_eval = np.linspace(0, 190, 1000)
sol = solve_ivp(rocket_dynamics, t_span, [0, 0], t_eval=t_eval)

# Данные из KSP
plt.plot(time_ksp_filtered, speed_ksp_filtered, '-b', linewidth=1.5, label="KSP", alpha=0.8)

# Математическая модель
v_res = np.sqrt(sol.y[0] ** 2 + sol.y[1] ** 2)
plt.plot(sol.t, v_res, 'r-', linewidth=2, label="мат. модель")

# Настройка графика
plt.title("Скорость ракеты")
plt.xlabel("Время (с)")
plt.ylabel("Скорость (м/с)")
plt.grid(True)
plt.legend()  # Добавляем легенду
plt.show()