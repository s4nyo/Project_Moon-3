import krpc
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep

# Подключение к серверу kRPC
conn = krpc.connect()

# Получение объекта космического корабля
vessel = conn.space_center.active_vessel

# Создание массивов для данных о времени и скорости
time_values = []
speed_values = []

# Получение скорости корабля на протяжении полета
while True:
    time = conn.space_center.ut
    speed = vessel.flight(vessel.orbit.body.reference_frame).speed
    time_values.append(time)
    speed_values.append(speed)
    print("Время: {}, Скорость корабля: {} м/с".format(time, speed))

    # Проверка условия завершения сбора данных
    altitude = vessel.flight().surface_altitude
    if altitude > 100000:
        break
    sleep(0.1)


df = pd.DataFrame({"speed": speed_values, "time": time_values})
df.to_excel('ksp_res3.xlsx')
# Построение графика скорости от времени
plt.plot(time_values, speed_values)
plt.title('Зависимость скорости от времени')
plt.xlabel('Время, s')
plt.ylabel('Скорость, m/s')
plt.show()