import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path

    """
    # Follow the pseudocode from lecture to complete greedy search :-)

    # return (forward_path, forward_prob)
    sen = []
    Terminalblank = False
    forward_prob = 1
    for t in range(y_probs.shape[1]): # seq
        max_index = np.argmax(y_probs[:, t, 0]) # index with maximum prob
        forward_prob *= max(y_probs[:, t, 0]) #  prob this step
        if max_index != 0:
            if Terminalblank:
                sen.append(SymbolSets[max_index - 1]) # blank
                Terminalblank = False
            else:
                if len(sen) == 0 or sen[-1] != SymbolSets[max_index - 1]: #
                    sen.append(SymbolSets[max_index - 1])
        else:
            Terminalblank = True
    forward_path = ''.join(sen)
    return forward_path, forward_prob


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    # global PathScore, BlankPathScore
    bs = beamSearch(SymbolSets, y_probs, BeamWidth)
    return bs.main()


class beamSearch():
    def __init__(self, SymbolSets, y_probs, BeamWidth):
        self.SymbolSets = SymbolSets
        self.y_probs = y_probs
        self.BeamWidth = BeamWidth
        self.PathScore = {}
        self.BlankPathScore = {}

    def main(self):
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(
            self.SymbolSets, self.y_probs[:, 0])
        for t in range(1, self.y_probs.shape[1]):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, self.BlankPathScore, self.PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, self.BeamWidth)
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.y_probs[:, t])
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.SymbolSets, self.y_probs[:, t])

        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        return self.bestPath(MergedPaths, FinalPathScore)

    # 1. InitializePaths
    def InitializePaths(self, SymbolSet, y):
        InitialBlankPathScore = {}
        InitialPathScore = {}
        InitialBlankPathScore[''] = y[0]
        InitialPathsWithFinalBlank = {''}
        InitialPathsWithFinalSymbol = set()
        for i, c in enumerate(SymbolSet):
            path = c
            InitialPathScore[path] = y[i + 1]
            InitialPathsWithFinalSymbol.add(path)
        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    # 2. Extending with blanks
    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = {}
        for path in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = self.BlankPathScore[path] * y[0]

        for path in PathsWithTerminalSymbol:
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += self.PathScore[path] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.add(path)
                UpdatedBlankPathScore[path] = self.PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    # 3. Extending with symbols
    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = {}

        for path in PathsWithTerminalBlank:
            for i, c in enumerate(SymbolSet):
                new_path = path + c
                UpdatedPathsWithTerminalSymbol.add(new_path)
                UpdatedPathScore[new_path] = self.BlankPathScore[path] * y[i + 1]
        for path in PathsWithTerminalSymbol:
            for i, c in enumerate(SymbolSet):
                new_path = path if c == path[-1] else path + c
                if new_path in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[new_path] += self.PathScore[path] * y[i + 1]
                else:
                    UpdatedPathsWithTerminalSymbol.add(new_path)
                    UpdatedPathScore[new_path] = self.PathScore[path] * y[i + 1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    # 4. Prune
    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}

        scorelist = []
        # print(BlankPathScore)
        #
        # print(PathsWithTerminalBlank)
        # print('=' * 50)
        for p in PathsWithTerminalBlank:
            scorelist += [BlankPathScore[p]]
        for p in PathsWithTerminalSymbol:
            scorelist += [PathScore[p]]
        # print("original")
        # print(scorelist)
        # print('='*40)
        scorelist.sort(reverse=True)
        # print("decreasing")
        # print(scorelist)

        cutoff = scorelist[BeamWidth - 1] if BeamWidth < len(scorelist) else scorelist[-1]

        PrunedPathsWithTerminalBlank = set()
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.add(p)
                PrunedBlankPathScore[p] = BlankPathScore[p]
        PrunedPathsWithTerminalSymbol = set()
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.add(p)
                PrunedPathScore[p] = PathScore[p]
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

    # 5. Merge
    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.add(p)
                FinalPathScore[p] = BlankPathScore[p]
        return MergedPaths, FinalPathScore

    def bestPath(self, MergedPaths, FinalPathScore):
        temp_path = list(MergedPaths)
        best_path = temp_path[0]
        best_score = FinalPathScore[best_path]
        for path in FinalPathScore:
            if (FinalPathScore[path] > best_score):
                best_path = path
                best_score = FinalPathScore[path]
        return best_path, FinalPathScore

