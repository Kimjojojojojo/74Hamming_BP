import numpy as np

class ConvolutionalEncoder:
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        self.memory = [0, 0, 0]  # 초기 메모리 상태

    def encode(self, bit):
        # 입력 비트를 메모리의 맨 앞에 추가
        self.memory.insert(0, bit)
        self.memory = self.memory[:4]  # 메모리는 최대 4비트 유지

        # G1과 G2를 사용하여 출력 계산
        output1 = (self.memory[0] * self.g1[0] + self.memory[1] * self.g1[1] + self.memory[2] * self.g1[2] + self.memory[3] * self.g1[3]) % 2
        output2 = (self.memory[0] * self.g2[0] + self.memory[1] * self.g2[1] + self.memory[2] * self.g2[2] + self.memory[3] * self.g2[3]) % 2

        return [output1, output2]

    def encode_sequence(self, bit_sequence):
        encoded_sequence = []
        for bit in bit_sequence:
            encoded_sequence.extend(self.encode(bit))
        return encoded_sequence

    def symbol_mapper(self, encoded_sequence):
        encoded_symbol = []
        for bit in encoded_sequence:
            if bit == 0:
                encoded_symbol.append(-1)
            else:
                encoded_symbol.append(1)
        return encoded_symbol
    def symbol_decoder(self, encoded_symbol):
        decoded_bits = []
        for sym in encoded_symbol:
            if sym < 0:
                decoded_bits.append(0)
            else:
                decoded_bits.append(1)
        return decoded_bits

    def noise_ch(self, encoded_symbol, sigma):
        encoded_noised_symbol = []
        for sym in encoded_symbol:
            noise = np.random.normal(0, sigma)
            y = sym + noise
            encoded_noised_symbol.append(y)

        return encoded_noised_symbol


# # (7, 5) 다항식을 사용한 Convolutional Encoder 예제
# g1 = [1, 1, 0, 1]  # 1101 (octal 7)
# g2 = [1, 1, 1, 1]  # 1111 (octal 5)
#
# encoder = ConvolutionalEncoder(g1, g2)
# input_sequence = [1, 0, 1, 1, 0]  # 예제 입력 비트 시퀀스
# encoded_sequence = encoder.encode_sequence(input_sequence)
#
# print("Input sequence:  ", input_sequence)
# print("Encoded sequence:", encoded_sequence)
