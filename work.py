import numpy as np

theta = np.loadtxt('Theta.txt')
cars = float(input("Сколько тысяч авто в населенном пункте? "))

# Прогноз: y = theta0 + theta1 * x
result = theta[0] + theta[1] * cars
print(f"Прогнозируемая прибыль СТО: {result:.2f} тыс. $")