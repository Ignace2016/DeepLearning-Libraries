import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        B, _ = target.shape
        totalLoss = np.zeros(B)
        # <---------------------------------------------

        for b in range(B):
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
            # <---------------------------------------------

            # -------------------------------------------->

            ctc = CTC(self.BLANK)
            target_trunc = target[b, 0:target_lengths[b]] # (batch_size, paddedtargetlen)
            logits_trunc = logits[0:input_lengths[b], b] # (seqlength, batch_size, len(Symbols))
            extSymbols, skipConnect = ctc.targetWithBlank(target_trunc)
            alpha = ctc.forwardProb(logits_trunc, extSymbols, skipConnect)
            beta = ctc.backwardProb(logits_trunc, extSymbols, skipConnect)
            gamma = ctc.postProb(alpha, beta) # (input_len, 2 * target_len + 1)
            for r in range(gamma.shape[1]):
                totalLoss[b] -= np.sum(gamma[:, r] * np.log(logits_trunc[:, extSymbols[r]])) # KL-divergence

            self.gammas.append(gamma)
            # <---------------------------------------------
        totalLoss = np.mean(totalLoss)

        return totalLoss

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: (seqlength, batch_size, len(Symbols))
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->

            
            gamma = self.gammas[b]
            ctc = CTC(self.BLANK)
            target_trunc = self.target[b, 0:self.target_lengths[b]]
            logits_trunc = self.logits[0:self.input_lengths[b], b]
            extSymbols, skipConnect = ctc.targetWithBlank(target_trunc)
            for r in range(gamma.shape[1]):
                dY[0:self.input_lengths[b], b, extSymbols[r]] -= gamma[:, r] / logits_trunc[:, extSymbols[r]]
            # <---------------------------------------------

        return dY
