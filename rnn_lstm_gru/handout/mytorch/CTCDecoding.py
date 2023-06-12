from collections import Counter

import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)


        symbol_length, seq_len, batch_size = y_probs.shape


        for t in range(seq_len):
            probs_t = y_probs[:, t, :]
            max_idx = np.argmax(probs_t)
            max_prob = probs_t[max_idx]
            if max_idx == 0:
                decoded_path.append('BLANK')
            else:
                decoded_path.append(self.symbol_set[max_idx - 1])
            path_prob *= max_prob

        compressed_path = [decoded_path[0]]
        for i in range(1, len(decoded_path)):
            if decoded_path[i] != decoded_path[i - 1]:
                compressed_path.append(decoded_path[i])

        compressed_path = np.array(compressed_path)
        decoded_path = "".join(compressed_path[compressed_path != 'BLANK'])


        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        num_symbols, seq_len, batch_size = y_probs.shape

        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(
            self.symbol_set, y_probs[:, 0, :])

        for t in range(1, seq_len):
            PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = self.Prune(
                NewPathsWithTerminalBlank,
                NewPathsWithTerminalSymbol,
                NewBlankPathScore, NewPathScore,
                self.beam_width)

            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank,
                                                                           PathsWithTerminalSymbol, y_probs[:, t, :],
                                                                           BlankPathScore, PathScore)

            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol,
                                                                        self.symbol_set, y_probs[:, t, :], BlankPathScore,
                                                                        PathScore)

        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
                                                          NewBlankPathScore, NewPathScore)

        print(FinalPathScore)
        bestPath = max(FinalPathScore, key=FinalPathScore.get)

        return bestPath, FinalPathScore


    def InitializePaths(self, SymbolSets, y):
        InitialBlankPathScore = dict()
        InitialPathScore = dict()


        path = ""
        InitialBlankPathScore[path] = y[0]
        InitialPathsWithFinalBlank = set()
        InitialPathsWithFinalBlank.add(path)

        InitialPathWithFinalSymbol = set()

        for i in range(len(SymbolSets)):
            path = SymbolSets[i]
            InitialPathScore[path] = y[i + 1]
            InitialPathWithFinalSymbol.add(path)

        return InitialPathsWithFinalBlank, InitialPathWithFinalSymbol, InitialBlankPathScore, InitialPathScore


    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = dict()

        for path in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.add(path)
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

        for path in PathsWithTerminalSymbol:
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y[0]
            else:
                UpdatedPathsWithTerminalBlank.add(path)
                UpdatedBlankPathScore[path] = PathScore[path] * y[0]

        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = dict()

        for path in PathsWithTerminalBlank:
            for i in range(len(SymbolSet)):
                newPath = path + SymbolSet[i]
                UpdatedPathsWithTerminalSymbol.add(newPath)
                UpdatedPathScore[newPath] = BlankPathScore[path] * y[i + 1]

        for path in PathsWithTerminalSymbol:
            for i in range(len(SymbolSet)):
                newPath = path if (SymbolSet[i] == path[-1]) else path + SymbolSet[i]
                if newPath in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[newPath] += PathScore[path] * y[i + 1]
                else:
                    UpdatedPathsWithTerminalSymbol.add(newPath)
                    UpdatedPathScore[newPath] = PathScore[path] * y[i + 1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore



    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore = dict()
        PrunedPathScore = dict()
        PrunedPathsWithTerminalBlank = set()
        PrunedPathsWithTerminalSymbol = set()

        scoreList = []


        for p in PathsWithTerminalBlank:
            scoreList.append(BlankPathScore[p])
        for p in PathsWithTerminalSymbol:
            scoreList.append(PathScore[p])


        scoreList.sort(reverse=True)
        cutoff = scoreList[BeamWidth] if (BeamWidth < len(scoreList)) else scoreList[-1]

                
                
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] > cutoff:
                PrunedPathsWithTerminalBlank.add(p)
                PrunedBlankPathScore[p] = BlankPathScore[p]
                            

        for p in PathsWithTerminalSymbol:
            if PathScore[p] > cutoff:
                PrunedPathsWithTerminalSymbol.add(p)
                PrunedPathScore[p] = PathScore[p]

        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore



    def MergeIdenticalPaths(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore

        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] = FinalPathScore[p] + BlankPathScore[p]
            else:
                MergedPaths.add(p)
                FinalPathScore[p] = BlankPathScore[p]
        return MergedPaths, FinalPathScore




