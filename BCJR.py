import numpy as np


class BCJRDecoder:
    def __init__(self, g1, g2):
        self.g1 = g1
        self.g2 = g2
        self.num_states = 2 ** (len(g1) - 1)

    def _get_next_state(self, current_state, input_bit): # define state transition
        current_state_bits = list(map(int, format(current_state, '0' + str(len(self.g1) - 1) + 'b')))
        next_state_bits = [input_bit] + current_state_bits[:-1] # if current state : 1 -> 110 = next state : 111
        next_state = int(''.join(map(str, next_state_bits)), 2) # binary to decimal
        return next_state

    def _get_output(self, state, input_bit):
        state_bits = list(map(int, format(state, '0' + str(len(self.g1) - 1) + 'b')))
        current_bits = [input_bit] + state_bits
        output1 = sum([current_bits[i] * self.g1[i] for i in range(len(self.g1))]) % 2
        output2 = sum([current_bits[i] * self.g2[i] for i in range(len(self.g2))]) % 2
        return [output1, output2]

    def decode(self, received_sequence, noise_variance):
        num_steps = len(received_sequence) // 2
        alpha = np.zeros((num_steps + 1, self.num_states))
        beta = np.zeros((num_steps + 1, self.num_states))
        gamma = np.zeros((num_steps, self.num_states, 2, self.num_states))

        alpha[0, 0] = 1
        beta[-1, :] = 1

        for k in range(num_steps):
            for s in range(self.num_states):
                for input_bit in [0, 1]:
                    next_state = self._get_next_state(s, input_bit)
                    output = self._get_output(s, input_bit)
                    received_output = received_sequence[2 * k:2 * (k + 1)]
                    p = np.exp(-np.sum((np.array(received_output) - np.array(output))**2) / (2 * noise_variance))
                    gamma[k, s, input_bit, next_state] = p

        for k in range(1, num_steps + 1):
            sum_tmp = 0
            for s in range(self.num_states):
                alpha[k, s] = sum(alpha[k - 1, prev_s] * gamma[k - 1, prev_s, input_bit, s]
                                  for prev_s in range(self.num_states)
                                  for input_bit in [0, 1])
                sum_tmp += alpha[k, s]
            alpha[k] = alpha[k] / sum_tmp

        for k in range(num_steps - 1, -1, -1):
            sum_tmp = 0
            for s in range(self.num_states):
                beta[k, s] = sum(beta[k + 1, next_s] * gamma[k, s, input_bit, next_s]
                                 for next_s in range(self.num_states)
                                 for input_bit in [0, 1])
                sum_tmp += beta[k, s]
            beta[k] = beta[k] / sum_tmp

        llr = np.zeros(num_steps)
        for k in range(num_steps):
            num = sum(alpha[k, s] * beta[k + 1, self._get_next_state(s, 1)] * gamma[k, s, 1, self._get_next_state(s, 1)]
                      for s in range(self.num_states))
            den = sum(alpha[k, s] * beta[k + 1, self._get_next_state(s, 0)] * gamma[k, s, 0, self._get_next_state(s, 0)]
                      for s in range(self.num_states))
            llr[k] = np.log(num / den)

        decoded_bits = (llr > 0).astype(int)
        return decoded_bits
