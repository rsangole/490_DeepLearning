import random
from math import exp
import numpy as np
import matplotlib.pyplot as py

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


def welcome():
    print 'This program learns to distinguish between five capital letters: X, M, H, A, and N'
    return ()

# Compute neuron activation using sigmoid transfer function
def computeTransferFnctn(summedNeuronInput, alpha):
    activation = 1.0 / (1.0 + exp(-alpha * summedNeuronInput))
    return activation

# Compute derivative of transfer function
def computeTransferFnctnDeriv(NeuronOutput, alpha):
    return alpha * NeuronOutput * (1.0 - NeuronOutput)

# Matrix dot product
def matrixDotProduct(matrx1, matrx2):
    dotProduct = np.dot(matrx1, matrx2)
    return (dotProduct)

# NN Size Specs
def obtainNeuralNetworkSizeSpecs(numInputNodes=25, numHiddenNodes=6, numOutputNodes=5):
    # This procedure operates as a function, as it returns a single value (which really is a list of
    #    three values). It is called directly from 'main.'
    #
    # This procedure allows the user to specify the size of the input (I), hidden (H),
    #    and output (O) layers.
    # These values will be stored in a list, the arraySizeList.
    # This list will be used to specify the sizes of two different weight arrays:
    #   - wWeights; the Input-to-Hidden array, and
    #   - vWeights; the Hidden-to-Output array.
    # However, even though we're calling this procedure, we will still hard-code the array sizes for now.
    print ' '
    print '  The number of nodes at each level are:'
    print '    Input: 5x5 (square array)'
    print '    Hidden: %d' %numHiddenNodes
    print '    Output: %d' %numOutputNodes

    return numInputNodes, numHiddenNodes, numOutputNodes

# Single weight randomly picked between -1 and 1
def InitializeWeight():
    randomNum = random.random()
    weight = 1 - 2 * randomNum      #returns a random number between -1 and 1
    return (weight)

# Weight array of needed size using lower & upper layer sizes. Called from main
def initializeWeightArray(weightArraySizeList):
    numBottomNodes = weightArraySizeList[0]
    numUpperNodes = weightArraySizeList[1]
    weightArray = np.zeros((numUpperNodes, numBottomNodes))  # initialize the weight matrix with 0's
    for row in range(numUpperNodes):
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        # For a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numBottomNodes):
            weightArray[row, col] = InitializeWeight()
    return (weightArray)

# Bias array. Called from main
def initializeBiasWeightArray(numBiasNodes):
    biasWeightArray = np.zeros(numBiasNodes)  # iniitalize the weight matrix with 0's
    for node in range(numBiasNodes):  # Number of nodes in bias weight set
        biasWeightArray[node] = InitializeWeight()
    return (biasWeightArray)

# Give a dataset #, get back a letter
def obtainSelectedAlphabetTrainingValues(trainingDataSetNum):
    if trainingDataSetNum == 0: trainingDataList = (
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1,
        0, 'X')
    if trainingDataSetNum == 1: trainingDataList = (
        1, 0, 0, 0, 1,
        1, 1, 0, 1, 1,
        1, 0, 1, 0, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 'M')
    if trainingDataSetNum == 2: trainingDataList = (
        1, 0, 0, 0, 1,
        1, 1, 0, 0, 1,
        1, 0, 1, 0, 1,
        1, 0, 0, 1, 1,
        1, 0, 0, 0, 1,
        2, 'N')
    if trainingDataSetNum == 3: trainingDataList = (
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        3, 'H')
    if trainingDataSetNum == 4: trainingDataList = (
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 1, 1, 1, 1,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        4, 'A')
    return (trainingDataList)

# Gives a random letter. Can specify how many letters to choose from.
def obtainRandomAlphabetTrainingValues(numTrainingDataSets):
    # 9 #s letter......... , output_class, letter_in_string
    trainingDataSetNum = random.randint(0, numTrainingDataSets)
    if trainingDataSetNum == 0: trainingDataList = (
    1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 'X')
    if trainingDataSetNum == 1: trainingDataList = (
    1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 'M')
    if trainingDataSetNum == 2: trainingDataList = (
    1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 'N')
    if trainingDataSetNum == 3: trainingDataList = (
    1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 3, 'H')
    if trainingDataSetNum == 4: trainingDataList = (
    0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 4, 'A')
    return (trainingDataList)

# Feed Forward


# Back Prop


# Main

if __name__ == "__main__": main()
