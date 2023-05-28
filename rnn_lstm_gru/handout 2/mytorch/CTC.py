import numpy as np

import loss


class CTC(object):

    def __init__(self, BLANK=0):
        """
		Initialize instance variables
		Argument(s)
		BLANK (int, optional): blank label index. Default 0.
		"""

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
		"""

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)


        N = len(extended_symbols)
        skip_connect = np.zeros_like(extended_symbols)

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        i = 2
        while i < N:
            j = i - 2
            if (extended_symbols[i] != self.BLANK) and (extended_symbols[i] != extended_symbols[j]):
                skip_connect[i] = 1
            i = i + 1

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        input_len, N = len(logits), len(extended_symbols)
        alpha = np.zeros(shape=(input_len, N))

        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]

        for t in range(1, input_len):
            alpha[t][0] = alpha[t - 1][0] * logits[t][extended_symbols[0]]
            for l in range(1, N):
                alpha[t][l] = alpha[t - 1][l] + alpha[t - 1][l - 1]
                if skip_connect[l]:
                    alpha[t][l] += alpha[t - 1][l - 2]
                alpha[t][l] *= logits[t][extended_symbols[l]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
		"""
        T, N = len(logits), len(extended_symbols)
        beta = np.zeros(shape=(T, N))
        beta[T - 1][N - 1] = logits[T - 1][extended_symbols[N - 1]]
        beta[T - 1][N - 2] = logits[T - 1][extended_symbols[N - 2]]

        for t in range(T - 2, -1, -1):
            beta[t][N - 1] = beta[t + 1][N - 1] * logits[t][extended_symbols[N - 1]]
            for i in range(N - 2, -1, -1):
                beta[t][i] = beta[t + 1][i] + beta[t + 1][i + 1]
                if (i <= N - 3) and (extended_symbols[i] != extended_symbols[i + 2]):
                    beta[t][i] += beta[t + 1][i + 2]
                beta[t][i] *= logits[t][extended_symbols[i]]

        for t in range(T - 1, -1, -1):
            for i in range(N - 1, -1, -1):
                beta[t][i] = beta[t][i] / logits[t][extended_symbols[i]]

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability
		"""

        [T, N] = alpha.shape
        gamma = np.zeros(shape=(T, N))
        sumgamma = np.zeros((T,))

        for t in range(T):
            for i in range(N):
                gamma[t][i] = alpha[t][i] * beta[t][i]
                sumgamma[t] += gamma[t][i]

            for i in range(N):
                gamma[t][i] = gamma[t][i] / sumgamma[t]

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """
		Initialize instance variables
        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
		"""
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.extended_symbols = []
        self.ctc = CTC()

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward
		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
		log probabilities (output sequence) from the RNN/GRU
		y_hat = logits[:i_len, batch_itr, :]
        target [np.array, dim=(batch_size, padded_target_len)]: target sequences

        input_lengths [np.array, dim=(batch_size,)]: lengths of the inputs
        target_lengths [np.array, dim=(batch_size,)]: lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target
        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        # self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result

            ctc = CTC()
            # Truncate the target to target length
            y_len = target_lengths[batch_itr]
            y = target[batch_itr, :y_len]

            # Truncate the logits to input length
            i_len = input_lengths[batch_itr]
            y_hat = logits[:i_len, batch_itr, :]

            # Extend target sequence with blank
            y_extend, skip_connect = ctc.extend_target_with_blank(y)

            # Compute forward probabilities
            alpha = ctc.get_forward_probs(y_hat, y_extend, skip_connect)

            # Compute backward probabilities
            beta = ctc.get_backward_probs(y_hat, y_extend, skip_connect)

            # Compute posteriors using total probability function
            prob_posterior = ctc.get_posterior_probs(alpha, beta)
            self.gammas.append(prob_posterior)
            self.extended_symbols.append(y_extend)

            T, r = y_hat.shape[0], len(y_extend)

            for t in range(T):
                for sys_r in range(r):
                    total_loss[batch_itr] += (prob_posterior[t, sys_r] * np.log(y_hat[t, y_extend[sys_r]]))

        total_loss = -np.sum(total_loss) / B

        return total_loss

    def backward(self):

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY

            # Truncate the target to target length

            y = self.extended_symbols[batch_itr]
            N = len(y)
            i_len = self.input_lengths[batch_itr]
            y_hat = self.logits[:i_len, batch_itr, :]
            prob_posterior = self.gammas[batch_itr]
            T = y_hat.shape[0]

            for t in range(T):
                for i in range(N):
                    dY[t, batch_itr, y[i]] -= (prob_posterior[t, i] / y_hat[t, y[i]])

        return dY
