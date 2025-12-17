import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants

m0 = 0  # масса без топлива
M = 0  # масса с топливом
Cf = 0.3  # коэффициент лобового сопротивления
ro = 1.293  # плотность воздуха
S = constants.pi * ((2.2 / 2) ** 2)  # площадь сечения
g = constants.g
F = [2649660, 860000] # силы тяги


# Функция для расчета угла наклона траектории γ(t) относительно горизонта
def gamma_angle(t):
    """
    Угол наклона траектории относительно горизонта:
    - t < 70: вертикальный взлет (90°)
    - 10 <= t < 140: поворот от 90° до 45°
    - t >= 140: плавный поворот от 45° до 5°
    """
    if t < 70:
        return np.radians(90)  # вертикальный взлет
    elif t < 140:
        # Линейный поворот от 90° до 45°
        angle = 90 - (t - 70) * 45 / 90
        return np.radians(angle)
    else:
        # Плавный поворот от 45° до 5°
        angle = 5 + 40 * np.exp(-(t - 140) / 150)
        return np.radians(max(angle, 5))


# Функция для расчета скорости с учетом гравитационных потерь
def dv_dt_with_gravity_loss(t, v):
    # Расчет гравитационной компоненты: g * sin(γ)
    # γ - угол относительно горизонта, поэтому используем sin
    gravity_component = g * np.sin(gamma_angle(t))

        if t < 88:
            M = 205900
            m0 = 109900
            Ft = F[0]
            k = (M - m0) / 88
            current_mass = M - k * t
            return ((Ft / current_mass) - ((Cf * ro * S) / (2 * current_mass)) * v ** 2 - gravity_component)
        elif t < 190:
            M = 67600
            m0 = 27700
            Ft = F[1]
            k = (M - m0) / 146
            t_local = t - 88
            current_mass = M - k * t_local
            return ((Ft / current_mass) - ((Cf * ro * S) / (2 * current_mass)) * v ** 2 - gravity_component)

v0 = 0
t_model = np.linspace(0, 210, 245)  # Увеличиваем количество точек для точности

solve_with_gravity = integrate.solve_ivp(
    dv_dt_with_gravity_loss,
    t_span=(0, max(t_model)),
    y0=[v0],
    t_eval=t_model,
    method='RK45',
    rtol=1e-6,
    atol=1e-9
)

# Построение графика скорости
plt.figure(figsize=(10, 6))
plt.plot(solve_with_gravity.t, solve_with_gravity.y[0], '-r', linewidth=2.5)
plt.title('Зависимость скорости ракеты от времени с учетом гравитационных потерь', fontsize=14)
plt.xlabel('Время, с', fontsize=12)
plt.ylabel('Скорость, м/с', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



