import numpy as np
import random
import matplotlib.pyplot as plt
from convolution import ConvolutionalEncoder
from BCJR import BCJRDecoder
from scipy.special import erfc

import detector
import Hamming

def sigma_from_db(db):
    return 10 ** (db / 20)

def error_probability(sigma):
    return 0.5 * erfc(1 / (np.sqrt(2) * sigma))

# Generate a numpy array from '0000' to '1111'
g1 = [1, 1, 0, 1]  # 1101 (octal 13)
g2 = [1, 1, 1, 1]  # 1111 (octal 15)

encoder = ConvolutionalEncoder(g1, g2)
decoder = BCJRDecoder(g1, g2)
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
db_range = np.arange(-10, 21, 1)  # -10dB부터 20dB까지 1dB 간격

rows = 100
cols = 4

error_rates_BP = []
error_rates_ML = []
error_rates_Hamming = []
error_rates_LBP = []

num_samples = 100

for db in db_range:
    sigma = sigma_from_db(db)
    error_sum_BP = 0
    error_sum_ML = 0
    error_sum_Hamming = 0
    error_sum_LBP = 0

    for n in range(num_samples):
        print("sample number",db, n)
        # 0과 1을 균등 분포로 가지는 100x4 배열 생성
        message_list = [random.choices([0, 1], k=cols) for _ in range(rows)]
        message_str_list = [''.join(map(str, message)) for message in message_list]
        codewords_list = []
        codewords_CH_list = []

        # message -> Hamming encoder
        for i in range(rows):
            code = ''.join(map(str, message_list[i]))
            codewords_list.append(Hamming.hamming_7_4_encoder(code))

        codewords_CH_list = np.zeros((rows, N))
        # print('b=',codewords_CH_list)
        # AWGN CH
        for r in range(rows):
            codeword = np.array(list(codewords_list[r])).astype(int)
            #print(codeword)
            encoded_symbol = encoder.symbol_mapper(codeword)
            noised_encoded_symbol = encoder.noise_ch(encoded_symbol, sigma)
            #print(noised_encoded_symbol)
            decoded_bits = encoder.symbol_decoder(noised_encoded_symbol)
            codewords_CH_list[r] = decoded_bits
        # print('aa',codewords_CH_list)

        # for r in range(rows):
        #     codeword = list(codewords_CH_list[r])  # 문자열을 리스트로 변환
        #     for n in range(N):
        #         random_number = random.random()
        #         if random_number < e:
        #             if codeword[n] == '1':
        #                 codeword[n] = '0'
        #             else:
        #                 codeword[n] = '1'
        #     codewords_CH_list[r] = ''.join(codeword)

        error_bits_BP = np.zeros(rows)
        error_bits_ML = np.zeros(rows)
        error_bits_Hamming = np.zeros(rows)
        error_bits_LBP = np.zeros(rows)

        e = error_probability(sigma)
        for r in range(rows):
            # print('len=', len(codewords_CH_list[r]))
            detected_BP = detector.BP_detector(codewords_CH_list[r], codewords, e)
            #detected_ML = detector.ML_detector(codewords_CH_list[r], e)
            #detected_LBP = detector.ML_detector(codewords_CH_list[r], e)

            decoded_BP = Hamming.hamming_7_4_decoder(detected_BP)
            #decoded_ML = Hamming.hamming_7_4_decoder(detected_ML)
            #decoded_Hamming = Hamming.hamming_7_4_decoder(codewords_CH_list[r])
            #decoded_LBP = Hamming.hamming_7_4_decoder(detected_LBP)


            error_bits_BP[r] = sum(1 for a, b in zip(message_str_list[r], decoded_BP) if a != b)
            #error_bits_ML[r] = sum(1 for a, b in zip(message_str_list[r], decoded_ML) if a != b)
            #error_bits_Hamming[r] = sum(1 for a, b in zip(message_str_list[r], decoded_Hamming) if a != b)
            #error_bits_LBP[r] = sum(1 for a, b in zip(message_str_list[r], decoded_LBP) if a != b)

        error_sum_BP += np.sum(error_bits_BP)
        #error_sum_ML += np.sum(error_bits_ML)
        #error_sum_Hamming += np.sum(error_bits_Hamming)
        #error_sum_LBP += np.sum(error_bits_LBP)

    avg_error_rate_BP = error_sum_BP / (num_samples * rows * cols)
    #avg_error_rate_ML = error_sum_ML / (num_samples * rows * cols)
    #avg_error_rate_Hamming = error_sum_Hamming / (num_samples * rows * cols)
    #avg_error_rate_LBP = error_sum_LBP / (num_samples * rows * cols)

    error_rates_BP.append(avg_error_rate_BP)
    #error_rates_ML.append(avg_error_rate_ML)
    #error_rates_Hamming.append(avg_error_rate_Hamming)
    #error_rates_LBP.append(avg_error_rate_LBP)

g1 = [1, 1, 0, 1]  # 1101 (octal 13)
g2 = [1, 1, 1, 1]  # 1111 (octal 15)

encoder = ConvolutionalEncoder(g1, g2)
decoder = BCJRDecoder(g1, g2)


# 64비트 랜덤 메시지 생성 (0과 1은 uniform 확률)
def generate_message(length):
    return np.random.choice([0, 1], size=length)


def sigma_from_db(db):
    return 10 ** (db / 20)


db_range = np.arange(-10, 21, 1)  # -10dB부터 20dB까지 1dB 간격
ber_list = []

message_length = 400

for idx ,db in enumerate(db_range):
    print('dB',db)
    sigma = sigma_from_db(db)
    ber_sum = 0
    for n in range(num_samples):
        print('sample number:',n )
        message = generate_message(message_length)
        encoded_bits = encoder.encode_sequence(message)
        encoded_symbol = encoder.symbol_mapper(encoded_bits)
        encoded_noised_symbol = encoder.noise_ch(encoded_symbol, sigma)

        decoded_bits = decoder.decode(encoded_noised_symbol, sigma)

        # 비트 오류율 계산
        bit_errors = np.sum(message != decoded_bits)
        ber = bit_errors / message_length
        ber_sum += ber

    ber_avg = ber_sum / num_samples
    ber_list.append(ber_avg)

# 그래프 그리기
plt.semilogy(db_range, error_rates_BP, marker='o', label='Linear code with LBP Detector')
plt.semilogy(db_range, ber_list, marker='o', label='Convolution code with BCJR Detector' )
#plt.semilogy(db_range, error_rates_ML, marker='s', label='ML Detector')
#plt.semilogy(db_range, error_rates_Hamming, marker='d', label='Only Hamming Decoder')
#plt.semilogy(db_range, error_rates_LBP, marker='h', label='LBP Decoder')
plt.xlabel('Error Probability ')
plt.ylabel('Bit Error Rate[dB]')
plt.title('Linear Decoder Comparison')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
