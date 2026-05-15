import numpy as np
import matplotlib.pyplot as plt
# 1. Создание матрицы
def warmUpExercise(n):
    return np.eye(n)

# 2. Показ исходных данных
def plotData(X, y):
    plt.figure(figsize=(10, 7))
    plt.plot(X, y, 'rx', markersize=10, label='Данные СТО')
    plt.xlabel("Кол-во автомобилей в городе (тыс.)")
    plt.ylabel("Прибыль СТО (тыс. $)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.title("Исходные данные")
    plt.show()

# Загрузка датасета
data = np.loadtxt('DataSet.txt', delimiter=',')
X = data[:, 0:1]
y = data[:, 1:2]

# Смотрим на точки
plotData(X, y)

# 3. Разбиение и подготовка матриц
# Мы делим данные, чтобы проверить модель на "честность", и добавляем столбец единиц x0=1,
# чтобы модель могла высчитать свободный коэффициент theta0.
def Train_test(X, y, ratio=0.7):
    m = len(y)
    indices = np.random.permutation(m)
    train_size = int(m * ratio)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = Train_test(X, y)

# Подготовка матриц
X_train_ext = np.append(np.ones((len(X_train), 1)), X_train, axis=1)
X_test_ext = np.append(np.ones((len(X_test), 1)), X_test, axis=1)

# 4. Функция стоимости
# Здесь мы реализуем расчет того, насколько наша прямая далека от реальных точек
def computeCost(X, y, theta):
    m = len(y)

    # Векторный расчет
    # h = X * theta (предсказание)
    # J = 1/(2m) * sum((h - y)^2)
    h = X @ theta
    error = h - y
    J = (1 / (2 * m)) * np.sum(np.square(error))
    return J

theta = np.zeros((2, 1))
print(f"Начальная ошибка: {computeCost(X_train_ext, y_train, theta):.4f}")

# 5. Градиентный спуск (Обучение)
def gradientDescent(X, y, alpha, theta, iterations):
    m = len(y)
    J_history = []
    for i in range(iterations):
        # Вычисляем градиент (направление спуска)
        gradient = (1/m) * (X.T @ (X @ theta - y))
        # Делаем шаг вниз
        theta = theta - alpha * gradient
        # Сохраняем историю ошибки
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# Параметры обучения
alpha = 0.01
iterations = 1500

# Обучаем
theta, J_history = gradientDescent(X_train_ext, y_train, alpha, theta, iterations)

print("Обучение завершено!")
print(f"Веса (theta): {theta.flatten()}")
print(f"Финальная ошибка: {J_history[-1]:.4f}")

# 6. Предсказание
# Теперь проверяем модель на тестовых данных, которые она не видела, и сохраним веса в файл.
def predict(X_test, y_test, theta):
    predictions = X_test @ theta
    mae = np.mean(np.abs(predictions - y_test))
    print(f"Средняя ошибка на тестах: {mae:.4f} тыс. $")
    return predictions

predict(X_test_ext, y_test, theta)

# 7. Сохранение
np.savetxt('Theta.txt', theta)
print("Веса записаны в Theta.txt")

# 8. Финальный график (Данные + Линия регрессии)
plt.figure(figsize=(10, 7))
plt.plot(X, y, 'rx', label='Реальные данные')

X_line = np.array([[min(X)[0]], [max(X)[0]]])
X_line_ext = np.append(np.ones((2, 1)), X_line, axis=1)
y_line = X_line_ext @ theta

plt.plot(X_line, y_line, 'b-', linewidth=2, label='Линейная регрессия')
plt.xlabel("Машины (тыс.)")
plt.ylabel("Прибыль ($)")
plt.title("Результат обучения")
plt.legend()
plt.grid(True)
plt.show()