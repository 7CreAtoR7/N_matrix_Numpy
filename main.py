import numpy as np

# На вход программы подается массив seq:
# ['AG----', 'AG----', 'AGA---', 'GGC---', 'AT----', 'GGG---']
# Нужно вывести 4 массива NumPy, где для каждой соответствующей
# буквы (сначала A, G, T, C) будет выводиться единичка, иначе ноль.
#
# Например для слоя А:
# [[[1. 0. 0. 0. 0. 0.] A = 1., G = 0., - = 0.
#   [1. 0. 0. 0. 0. 0.]
#   [1. 0. 1. 0. 0. 0.] A = 1., G = 0., A = 1., -=0.
#   [0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0.]]

"""В списке seq все слои"""
seq = ['AG----', 'AG----', 'AGA---', 'GGC---', 'AT----', 'GGG---']

"""функция выводит в одной матрице все слои по отдельности"""
massiv_A = [[0] * len(seq) for _ in range(len(seq))]
massiv_G = [[0] * len(seq) for _ in range(len(seq))]
massiv_T = [[0] * len(seq) for _ in range(len(seq))]
massiv_C = [[0] * len(seq) for _ in range(len(seq))]
matrix = []


def to_matrix(seq):
    count = 1
    for i in range(len(seq)):
        index = 0
        for letter in seq[i]:
            if count == 1:  # т.е. буква А
                if letter == 'A':
                    massiv_A[i][index] = 1
                else:
                    massiv_A[i][index] = 0
            index += 1
    matrix.append(massiv_A)
    count = 2
    for i in range(len(seq)):
        index = 0
        for letter in seq[i]:
            if count == 2:  # т.е. буква G
                if letter == 'G':
                    massiv_G[i][index] = 1
                else:
                    massiv_G[i][index] = 0
            index += 1
    matrix.append(massiv_G)
    count = 3
    for i in range(len(seq)):
        index = 0
        for letter in seq[i]:
            if count == 3:  # т.е. буква T
                if letter == 'T':
                    massiv_T[i][index] = 1
                else:
                    massiv_T[i][index] = 0
            index += 1
    matrix.append(massiv_T)
    count = 4
    for i in range(len(seq)):
        index = 0
        for letter in seq[i]:
            if count == 4:  # т.е. буква C
                if letter == 'C':
                    massiv_C[i][index] = 1
                else:
                    massiv_C[i][index] = 0
            index += 1
    matrix.append(massiv_C)

    a = np.asarray(matrix, dtype=np.float32)
    # трёхмерная матрица
    print(a)


"""Функция на вход принимает троичную матрицу"""


# [[[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
# [[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
# [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
# [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]
def from_matrix(m):
    result = []
    count = 0
    count_letter = 0
    for i in range(len(m[0])):
        one_line_list = []
        for lst in m:  # lst[i] = [1, 0, 0, 0, 0, 0]
            line = ['' for _ in range(len(m[0]))]
            count_letter += 1  # 1 - значит первая строка (рассматриваем букву А)

            for ind, symbol in enumerate(lst[i]):
                if symbol == 1 and count_letter == 1:
                    line[ind] = 'A'
                elif symbol == 1 and count_letter == 2:
                    line[ind] = 'G'
                elif symbol == 1 and count_letter == 3:
                    line[ind] = 'T'
                elif symbol == 1 and count_letter == 4:
                    line[ind] = 'C'
                else:
                    line[ind] = '-'
            if count_letter == 4:
                count_letter = 0
            one_line_list.append(line)
        result.append(''.join(map(max, zip(*one_line_list))))
        count += 1
    print(result)


to_matrix(seq)
print('====================Разделитель====================')
from_matrix(matrix)
