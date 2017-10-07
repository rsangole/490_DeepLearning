from math import exp
import numpy as np
import random
import matplotlib.pyplot as plt

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
    # print '  The number of nodes at each level are:'
    # print '    Input: 5x5 (square array)'
    # print '    Hidden: %d' %numHiddenNodes
    # print '    Output: %d' %numOutputNodes
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
# Input -> Hidden . Gives Hidden Array activations
def ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList, wWeightArray,
                                          biasHiddenWeightArray):
    # inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    # outputArrayLength = arraySizeList[2]

    # initialize the sum of inputs into the hidden array with 0's
    sumIntoHiddenArray = np.zeros(hiddenArrayLength)
    hiddenArray = np.zeros(hiddenArrayLength)

    sumIntoHiddenArray = matrixDotProduct(wWeightArray, inputDataList)

    for node in range(hiddenArrayLength):  # Number of hidden nodes
        hiddenNodeSumInput = sumIntoHiddenArray[node] + biasHiddenWeightArray[node]
        hiddenArray[node] = computeTransferFnctn(hiddenNodeSumInput, alpha)

    return (hiddenArray)

# Hidden -> Output . Gives output array activations
def ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hiddenArray, vWeightArray,
                                           biasOutputWeightArray):
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    # initialize the sum of inputs into the hidden array with 0's
    sumIntoOutputArray = np.zeros(hiddenArrayLength)
    outputArray = np.zeros(outputArrayLength)

    sumIntoOutputArray = matrixDotProduct(vWeightArray, hiddenArray)

    for node in range(outputArrayLength):  # Number of hidden nodes
        outputNodeSumInput = sumIntoOutputArray[node] + biasOutputWeightArray[node]
        outputArray[node] = computeTransferFnctn(outputNodeSumInput, alpha)

    return (outputArray)

# For given w and v wts, calculate SSE for all possible input characters to get total_SSE
def ComputeOutputsAcrossAllTrainingData(alpha, arraySizeList, numTrainingDataSets, wWeightArray,
                                        biasHiddenWeightArray, vWeightArray, biasOutputWeightArray):
    selectedTrainingDataSet = 0
    inputArrayLength = arraySizeList[0]
    # hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    while selectedTrainingDataSet < numTrainingDataSets + 1:
        trainingDataList = obtainSelectedAlphabetTrainingValues(selectedTrainingDataSet)
        inputDataList = []
        for node in range(inputArrayLength):
            trainingData = trainingDataList[node]
            inputDataList.append(trainingData)

        print ' >In ComputeOutputsAcrossAllTrainingData():'
        print '   >Data Set Number', selectedTrainingDataSet, 'for letter ', trainingDataList[26]

        hiddenArray = ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList, wWeightArray,
                                                            biasHiddenWeightArray)

        # print '   >The hidden node activations are:'
        # print hiddenArray

        outputArray = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hiddenArray, vWeightArray,
                                                             biasOutputWeightArray)

        print '   >The output node activations are:', outputArray

        desiredOutputArray = np.zeros(outputArrayLength)  # initialize the output array with 0's
        desiredClass = trainingDataList[25]  # identify the desired class
        desiredOutputArray[desiredClass] = 1  # set the desired output for that class to 1

        print '   >The desired output array values are: ', desiredOutputArray

        # Determine the error between actual and desired outputs

        # Initialize the error array
        errorArray = np.zeros(outputArrayLength)

        newSSE = 0.0
        for node in range(outputArrayLength):  # Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node] * errorArray[node]

        print '   >The error values are:', errorArray

        # Print the Summed Squared Error
        print '   >New SSE = %.6f' % newSSE

        selectedTrainingDataSet = selectedTrainingDataSet + 1

# Back Prop
# Wt changes on Hidden -> Output layer
def BackpropagateOutputToHidden(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):
    # Unpack array lengths
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    transferFuncDerivArray = np.zeros(outputArrayLength)  # initialize an array for the transfer function

    for node in range(outputArrayLength):  # Number of hidden nodes
        transferFuncDerivArray[node] = computeTransferFnctnDeriv(outputArray[node], alpha)


    deltaVWtArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the new hidden weights

    for row in range(outputArrayLength):  # Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes,
        #    and the columns correspond to the number of hidden nodes,
        #    which can be multiplied by the hidden node array (expressed as a column).
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt = -errorArray[row] * transferFuncDerivArray[row] * hiddenArray[col]
            deltaVWtArray[row, col] = -eta * partialSSE_w_V_Wt
            newVWeightArray[row, col] = vWeightArray[row, col] + deltaVWtArray[row, col]

    return (newVWeightArray)

# Bias on output layer
def BackpropagateBiasOutputWeights(alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):

    # The equation for the actual dependence of the Summed Squared Error on a given bias-to-output
    #   weight biasOutput(o) is:
    #   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
    # The transfer function derivative (transFuncDeriv) returned from computeTransferFnctnDeriv is given as:
    #   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
    # Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
    #   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
    #   The parameter alpha is included in transFuncDeriv

    # Unpack the output array length
    outputArrayLength = arraySizeList[2]

    deltaBiasOutputArray = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray = np.zeros(outputArrayLength)  # initialize an array for the new output bias weights
    transferFuncDerivArray = np.zeros(outputArrayLength)  # iniitalize an array for the transfer function

    for node in range(outputArrayLength):  # Number of hidden nodes
        transferFuncDerivArray[node] = computeTransferFnctnDeriv(outputArray[node], alpha)

    for node in range(outputArrayLength):  # Number of nodes in output array (same as number of output bias nodes)
        partialSSE_w_BiasOutput = -errorArray[node] * transferFuncDerivArray[node]
        deltaBiasOutputArray[node] = -eta * partialSSE_w_BiasOutput
        newBiasOutputWeightArray[node] = biasOutputWeightArray[node] + deltaBiasOutputArray[node]

    return (newBiasOutputWeightArray)

# Wt changes on Input -> Hidden layer
def BackpropagateHiddenToInput(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
                               inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray):
    # Unpack array lengths
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv

    for node in range(hiddenArrayLength):  # Number of hidden nodes
        transferFuncDerivHiddenArray[node] = computeTransferFnctnDeriv(hiddenArray[node], alpha)

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    transferFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    weightedErrorArray = np.zeros(hiddenArrayLength)  # initialize array

    for outputNode in range(outputArrayLength):  # Number of output nodes
        transferFuncDerivOutputArray[outputNode] = computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode] * transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  # Number of output nodes
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode] \
                                             + vWeightArray[outputNode, hiddenNode] * errorTimesTFuncDerivOutputArray[
                outputNode]

    deltaWWtArray = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray = np.zeros(
        (hiddenArrayLength, inputArrayLength))  # initialize an array for the new input-to-hidden weights

    for row in range(hiddenArrayLength):  # Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts = -transferFuncDerivHiddenArray[row] * inputArray[col] * weightedErrorArray[row]
            deltaWWtArray[row, col] = -eta * partialSSE_w_W_Wts
            newWWeightArray[row, col] = wWeightArray[row, col] + deltaWWtArray[row, col]

    return (newWWeightArray)

# Bias on hidden layer
def BackpropagateBiasHiddenWeights(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
                                   inputArray, vWeightArray, wWeightArray, biasHiddenWeightArray,
                                   biasOutputWeightArray):
    # Unpack array lengths
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    transferFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    weightedErrorArray = np.zeros(hiddenArrayLength)  # initialize array

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv
    partialSSE_w_BiasHidden = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights

    for node in range(hiddenArrayLength):  # Number of hidden nodes
        transferFuncDerivHiddenArray[node] = computeTransferFnctnDeriv(hiddenArray[node], alpha)

    for outputNode in range(outputArrayLength):  # Number of output nodes
        transferFuncDerivOutputArray[outputNode] = computeTransferFnctnDeriv(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode] * transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  # Number of output nodes
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode] + \
                                             vWeightArray[outputNode, hiddenNode] * errorTimesTFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):  # Number of rows in input-to-hidden weightMatrix
        partialSSE_w_BiasHidden[hiddenNode] = -transferFuncDerivHiddenArray[hiddenNode] * weightedErrorArray[hiddenNode]
        deltaBiasHiddenArray[hiddenNode] = -eta * partialSSE_w_BiasHidden[hiddenNode]
        newBiasHiddenWeightArray[hiddenNode] = biasHiddenWeightArray[hiddenNode] + deltaBiasHiddenArray[hiddenNode]

    return (newBiasHiddenWeightArray)

# The MAIN module comprising of calls to:
#   (1) Welcome
#   (2) Obtain neural network size specifications for a three-layer network consisting of:
#       - Input layer
#       - Hidden layer
#       - Output layer
#   (3) Initialize connection weight values
#       - w: Input-to-Hidden nodes
#       - v: Hidden-to-Output nodes
#   (4) Compute a feedforward pass in two steps
#       - Randomly select a single training data set
#       - Input-to-Hidden
#       - Hidden-to-Output
#       - Compute the error array
#       - Compute the new Summed Squared Error (SSE)
#   (5) Perform a single backpropagation training pass
def main(alpha = 1.0, eta = 0.5, maxNumIterations = 5000, epsilon = 0.05, numTrainingDataSets = 4, seed_value = 1):
    random.seed(seed_value)

    welcome()

    iteration = 0

    print '1. Setting up network...'
    # Obtain the actual sizes for each layer of the network
    arraySizeList = obtainNeuralNetworkSizeSpecs()

    # Unpack the list; ascribe the various elements of the list to the sizes of different network layers
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]
    print '....ip nodes: %d, hidden nodes: %d, output nodes: %d' %(inputArrayLength, hiddenArrayLength, outputArrayLength)
    print '....Done'

    # Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
    # The wWeightArray is for Input-to-Hidden
    # The vWeightArray is for Hidden-to-Output

    print '2. Initializing 1st random weights...'
    wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)
    vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)
    biasHiddenWeightArraySize = hiddenArrayLength
    biasOutputWeightArraySize = outputArrayLength

    # The node-to-node connection weights are stored in a 2-D array
    wWeightArray = initializeWeightArray(wWeightArraySizeList)
    vWeightArray = initializeWeightArray(vWeightArraySizeList)

    # The bias weights are stored in a 1-D array
    biasHiddenWeightArray = initializeBiasWeightArray(biasHiddenWeightArraySize)
    biasOutputWeightArray = initializeBiasWeightArray(biasOutputWeightArraySize)
    print '....Done'

    # Before we start training, get a baseline set of outputs, errors, and SSE
    print '3. Baseline output before any training...'
    ComputeOutputsAcrossAllTrainingData(alpha, arraySizeList, numTrainingDataSets, wWeightArray,
                                        biasHiddenWeightArray, vWeightArray, biasOutputWeightArray)
    print '....Done'

    print '4. Initiating trackers...'
    # Variables to track changes in wWeight, vWeight, hiddenBias, outputBias, and SSE
    wWeightTracker = dict()
    vWeightTracker = dict()
    hiddenBiasTracker = dict()
    outputBiasTracker = dict()
    SSETracker = dict()
    letterTracker = dict()
    print '....Done'

    # Next step - Obtain a single set of randomly-selected training values for alpha-classification
    print '5. Starting convergence iterations...'
    while iteration < maxNumIterations:

        print '  >> Iteration number: ', iteration

        # Increment the iteration count
        iteration = iteration + 1

        # For any given pass, we re-initialize the training list
        inputDataList = []

        # Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
        trainingDataList = obtainRandomAlphabetTrainingValues(numTrainingDataSets)

        for node in range(inputArrayLength):
            trainingData = trainingDataList[node]
            inputDataList.append(trainingData)

        print '     Training set randomly chosen: Letter ', trainingDataList[26]

        desiredOutputArray = np.zeros(outputArrayLength)  # initialize the output array with 0's
        desiredClass = trainingDataList[25]  # identify the desired class
        desiredOutputArray[desiredClass] = 1  # set the desired output for that class to 1

        print '     The desired output array values are: ', desiredOutputArray

        # Compute a single feed-forward pass and obtain the Actual Outputs

        hiddenArray = ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList,
                                                        wWeightArray, biasHiddenWeightArray)

        outputArray = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hiddenArray,
                                                             vWeightArray, biasOutputWeightArray)

            #  Optional alternative code for later use:
            #  Assign the hidden and output values to specific different variables
            #    for node in range(hiddenArrayLength):
            #        actualHiddenOutput[node] = actualAllNodesOutputList [node]
            #    for node in range(outputArrayLength):
            #        actualOutput[node] = actualAllNodesOutputList [hiddenArrayLength + node]

        # Initialize the error array
        errorArray = np.zeros(outputArrayLength)

        # Determine the error between actual and desired outputs
        newSSE = 0.0
        for node in range(outputArrayLength):  # Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node] * errorArray[node]

        # Perform backpropagation
        # Perform first part of the backpropagation of weight changes
        newVWeightArray = BackpropagateOutputToHidden(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
                                                      vWeightArray)
        newBiasOutputWeightArray = BackpropagateBiasOutputWeights(alpha, eta, arraySizeList, errorArray, outputArray,
                                                                  biasOutputWeightArray)

        # Perform first part of the backpropagation of weight changes
        newWWeightArray = BackpropagateHiddenToInput(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray,
                                                     inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray,
                                                     biasOutputWeightArray)
        newBiasHiddenWeightArray = BackpropagateBiasHiddenWeights(alpha, eta, arraySizeList, errorArray, outputArray,
                                                                  hiddenArray,
                                                                  inputDataList, vWeightArray, wWeightArray,
                                                                  biasHiddenWeightArray, biasOutputWeightArray)

        # Assign new values to the weight matrices
        vWeightArray = newVWeightArray[:]
        biasOutputWeightArray = newBiasOutputWeightArray[:]
        wWeightArray = newWWeightArray[:]
        biasHiddenWeightArray = newBiasHiddenWeightArray[:]

        # Compute a forward pass, test the new SSE
        hiddenArray = ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList,
                                                            wWeightArray, biasHiddenWeightArray)
        outputArray = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hiddenArray,
                                                             vWeightArray, biasOutputWeightArray)

        # Determine the error between actual and desired outputs
        newSSE = 0.0
        for node in range(outputArrayLength):  # Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node] * errorArray[node]

        print '     {*} For iteration %d, the SSE = %.6f' %(iteration, newSSE)

        # Tracker functionality
        wWeightTracker[iteration] = wWeightArray
        vWeightTracker[iteration] = vWeightArray
        hiddenBiasTracker[iteration] = biasHiddenWeightArray
        outputBiasTracker[iteration] = biasOutputWeightArray
        SSETracker[iteration] = newSSE
        letterTracker[iteration] = trainingDataList[26]

        if newSSE < epsilon:
            break

    print '\n** Out of while loop at iteration **', iteration

    # After training, get a new comparative set of outputs, errors, and SSE
    print ' '
    print '  After training:'
    ComputeOutputsAcrossAllTrainingData(alpha, arraySizeList, numTrainingDataSets, wWeightArray,
                                        biasHiddenWeightArray, vWeightArray, biasOutputWeightArray)

    return vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker

# if __name__ == "__main__": main()

vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker = main(alpha=1.0,
     eta=0.5,
     maxNumIterations=100,
     epsilon=0.1,
     numTrainingDataSets=4,
     seed_value=1
     )


def plotSSE(SSETracker, letterTracker):
    x, y = zip(*SSETracker.items())

    fig, ax = plt.subplots()
    im = ax.plot(x,y)
    ax.set_title('Total SSE over convergence')
    plt.show()

    return

