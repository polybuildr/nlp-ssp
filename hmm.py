import numpy as np

class HMM:
    def __init__(self, p_initial, p_transition, p_emission):
        self.p_initial = p_initial
        self.p_transition = p_transition
        self.p_emission = p_emission
        self.STATES = list(self.p_initial.keys())

    def viterbi(self, input_tokens):
        U = self.p_initial
        T = self.p_transition
        O = self.p_emission
        STATES = self.STATES
        num_states = len(STATES)
        viterbi_matrix = np.zeros([num_states, len(input_tokens)])
        path_matrix = np.zeros([num_states, len(input_tokens)], dtype='int')

        r = 0
        c = 0

        # seed first column
        first_token = input_tokens[0]
        for r in range(0, num_states):
            state = STATES[r]
            viterbi_matrix[r][0] = U[state] * O[state][first_token]

        for c in range(1, len(input_tokens)):
            for r in range(0, num_states):
                current_state = STATES[r]
                max_p = -1
                most_likely_prev_state_idx = None
                for prev_state_idx in range(0, num_states):
                    transition_probability = T[STATES[prev_state_idx]][current_state]
                    # account for probability of being in prev_state
                    p = transition_probability * viterbi_matrix[prev_state_idx][c - 1]
                    if p > max_p:
                        max_p = p
                        most_likely_prev_state_idx = prev_state_idx

                viterbi_matrix[r][c] = max_p * O[current_state][input_tokens[c]]
                path_matrix[r][c] = most_likely_prev_state_idx

        # get best row from last column
        best_end_idx = np.argmax([viterbi_matrix[r][len(input_tokens) - 1] for r in range(0, num_states)])

        best_states = []
        best_r = best_end_idx
        for c in range(len(input_tokens) - 1, -1, -1):
            best_states.append(STATES[best_r])
            best_r = path_matrix[best_r][c]

        best_states.reverse()
        return best_states
