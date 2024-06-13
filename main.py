import numpy as np
import random

import detector
import Hamming

# Generate a numpy array from '0000' to '1111'
k = 4
N = 7
start = int('0000', 2)
end = int('1111', 2) + 1

binary_array = np.array([np.binary_repr(i, width=4) for i in range(start, end)])

codewords = []
for i in range(2**k):
    codewords.append(Hamming.hamming_7_4_encoder(binary_array[i]))

codewords_array = np.array(codewords)

e_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

rows = 100
cols = 4

M = 100 # sample number


# 0과 1을 균등 분포로 가지는 100x4 배열 생성
message_list = [random.choices([0, 1], k=cols) for _ in range(rows)]
message_str_list = [''.join(map(str, message)) for message in message_list]
codewords_list = []
codewords_CH_list = []

# message -> Hamming encoder
for i in range(rows):
    code = ''.join(map(str, message_list[i]))
    codewords_list.append(Hamming.hamming_7_4_encoder(code))

codewords_CH_list = codewords_list.copy()

# CH
e = 0.5
for r in range(rows):
    codeword = list(codewords_CH_list[r])  # 문자열을 리스트로 변환
    for n in range(N):
        random_number = random.random()
        if random_number < e:
            if codeword[n] == '1':
                codeword[n] = '0'
            else:
                codeword[n] = '1'
    codewords_CH_list[r] = ''.join(codeword)

error_bits_BP = np.zeros(rows)
error_bits_ML = np.zeros(rows)

for r in range(rows):
    detected_BP = detector.BP_detector(codewords_CH_list[r], codewords, e)
    detected_ML = detector.ML_detector(codewords_CH_list[r], e)

    decoded_BP = Hamming.hamming_7_4_decoder(detected_BP)
    decoded_ML = Hamming.hamming_7_4_decoder(detected_ML)

    # 디코딩된 결과의 길이가 4인지 확인
    if len(decoded_BP) != cols or len(decoded_ML) != cols:
        raise ValueError(f"Decoded result has incorrect length. BP: {len(decoded_BP)}, ML: {len(decoded_ML)}")

    error_bits_BP[r] = sum(1 for a, b in zip(message_str_list[r], decoded_BP) if a != b)
    error_bits_ML[r] = sum(1 for a, b in zip(message_str_list[r], decoded_ML) if a != b)

    print('message:', message_list[r])
    print('BP:', decoded_BP, 'error:', error_bits_BP[r])
    print('ML:', decoded_ML, 'error:', error_bits_ML[r])

error_sum_BP = np.sum(error_bits_BP)
error_sum_ML = np.sum(error_bits_ML)

error_rate_BP = error_sum_BP / (rows*cols)
error_rate_MLP = error_sum_ML / (rows*cols)
print("Error bits BP:", error_bits_BP)
print("Error bits ML:", error_bits_ML)

print("Error rate BP:", error_rate_BP)
print("Error rate ML:", error_rate_MLP)