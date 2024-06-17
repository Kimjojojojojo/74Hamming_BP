import numpy as np
import matplotlib.pyplot as plt
from convolution import ConvolutionalEncoder
from BCJR import BCJRDecoder

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

num_samples = 1000
message_length = 64

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
plt.figure()
plt.plot(db_range, ber_list, marker='o')
plt.xlabel('$\sigma^2$ [dB]')
plt.ylabel('Bit Error Rate[dB]')
plt.ylim(1e-2, 1)
plt.title('BCJR Decoder')
plt.grid(True)
plt.yscale('log')
plt.show()
