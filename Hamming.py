import numpy as np


def hamming_7_4_encoder(data):
    """
    Encodes 4-bit data into 7-bit Hamming code.
    :param data: 4-bit binary string (e.g., '1011')
    :return: 7-bit Hamming code binary string (e.g., '1011010')
    """
    if len(data) != 4 or not set(data).issubset({'0', '1'}):
        raise ValueError("Data must be a 4-bit binary string.")

    G = np.array([[1, 1, 0, 1],
                  [1, 0, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 1, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    data_vec = np.array(list(map(int, data)))
    codeword = np.dot(G, data_vec) % 2
    return ''.join(map(str, codeword))


def hamming_7_4_decoder(codeword):
    """
    Decodes 7-bit Hamming code to 4-bit data, correcting a single bit error if present.
    :param codeword: 7-bit binary string (e.g., '1011010')
    :return: Corrected 4-bit binary string (e.g., '1011')
    """
    if len(codeword) != 7 or not set(codeword).issubset({'0', '1'}):
        raise ValueError("Codeword must be a 7-bit binary string.")

    H = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]])

    codeword_vec = np.array(list(map(int, codeword)))
    syndrome = np.dot(H, codeword_vec) % 2
    syndrome_value = int(''.join(map(str, syndrome)), 2)

    if syndrome_value != 0:
        # print(f"Error detected at position {syndrome_value}")
        codeword_vec[syndrome_value - 1] ^= 1  # Correct the error

    data_indices = [2, 4, 5, 6]
    data = codeword_vec[data_indices]
    return ''.join(map(str, data))
