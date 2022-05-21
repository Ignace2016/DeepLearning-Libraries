import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
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
        extSymbols = []
        skipConnect = []

        # -------------------------------------------->

        # Your Code goes here
        for i, ch in enumerate(target):
            extSymbols.append(self.BLANK)
            skipConnect.append(False)
            extSymbols.append(ch)
            skipConnect.append(target[i] != target[i - 1] and i > 0) # i should also > 0
        extSymbols.append(self.BLANK)
        skipConnect.append(False)
        extSymbols = np.array(extSymbols)
        skipConnect = np.array(skipConnect)
        # <---------------------------------------------
        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        alpha[0,0] = logits[0, extSymbols[0]]
        alpha[0,1] = logits[0, extSymbols[1]]
        for t in range(1, T):
            alpha[t,0] = alpha[t-1,0] * logits[t,extSymbols[0]]
            for i in range(1,S):
                alpha[t,i] = alpha[t-1,i-1] + alpha[t-1,i]
                if (skipConnect[i]):
                    alpha[t,i] += alpha[t-1,i-2]
                alpha[t,i] *= logits[t,extSymbols[i]]
        # <---------------------------------------------

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, len(Symbols)))
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
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))

        beta[T - 1, S - 1] = 1
        beta[T - 1, S - 2] = 1
        beta[T - 1, 1: S - 2 - 1] = 0
        for t in range(T - 2, -1, -1):
            beta[t, S - 1] = beta[t + 1, S - 1] * logits[t + 1, extSymbols[S - 1]]
            for i in range(S - 2, -1, -1):
                beta[t, i] = beta[t + 1, i] * logits[t + 1, extSymbols[i]] + beta[t + 1, i + 1] * logits[t + 1, extSymbols[i + 1]]
                if (i < S - 2 and skipConnect[i + 2]):
                    beta[t, i] += beta[t + 1, i + 2] * logits[t + 1, extSymbols[i + 2]]
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
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
        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))


        for t in range(T):
            sum_gamma_t = 0
            for i in range(S):
                gamma[t, i] = alpha[t, i] * beta[t, i]
                sum_gamma_t += gamma[t, i]
            for i in range(S):
                gamma[t, i] /= sum_gamma_t

        # <---------------------------------------------

        return gamma
