import numpy as np

# Ввести значения 0 или 1(False/True)
vodka = 0.0
rain = 0.0
friend = 0.0
music = 0.0
money = 1.0


def activation_function(x):
    if x >= 0.5:
        return 1
    else:
        return 0


def predict(vodka, rain, friend, music, mony):
    inpute = np.array([vodka, rain, friend, music, money])
    input_hidden_1 = [0.10, -1, 0.5, 0.5, 1]
    input_hidden_2 = [0.10, 0.5, 0.5, 0.5, -1]
    input_hidden_3 = [0.10, -1, 0.5, -0.25, 1]
    input_hidden_4 = [0.5, 0.5, 0.0, -0.25, 1]
    input_hidden_5 = [0.10, 0.5, 0.0, 0.5, 1]
    weight_input_hidden = np.array([input_hidden_1, input_hidden_2, input_hidden_3, input_hidden_4, input_hidden_5])

    hidden_output = np.array([1, -1, -0.5, 0.5, 1])

    matrix_hidden_input = np.dot(weight_input_hidden, inpute)
    print('Значения нейронов: ' + str(matrix_hidden_input))

    matrix_hidden_output = np.array([activation_function(x) for x in matrix_hidden_input])
    print('Значение 2 слоя нейронов: ' + str(matrix_hidden_output))

    final_output = np.dot(hidden_output, matrix_hidden_output)
    print('Выходное значение: ' + str(final_output))
    return activation_function(final_output) == 1


predict(vodka, rain, friend, music, money)