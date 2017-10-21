import sys
sys.path.append("/Users/Rahul/Documents/OneDrive/MSPA/490/Github/Week5_9x9Variation")
from week5_9x9Variation_RSangole import *

letter_list = [obtainSelectedAlphabetTrainingValues(x)[3] for x in range(25)]
class_list = set([obtainSelectedAlphabetTrainingValues(x)[5] for x in range(25)])

# Question 1:
# Get a convergent solution for all letters:
alpha = 1
eta = 1
maxNumIterations = 20000
epsilon = 0.01
numTrainingDataSets = 25
seed_value = 244
numH = 4
vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker, errors, classTracker = main(
    alpha=alpha,
    eta=eta,
        maxNumIterations=maxNumIterations,
    epsilon=epsilon,
    numTrainingDataSets=numTrainingDataSets,
    seed_value=seed_value,
    numHiddenNodes=numH
)
plotSSEbyClass(SSETracker, classTracker, alpha=alpha,
        eta=eta,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        errors=errors,
        numHiddenNodes=numH,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value)
plotSSEbyLetter(alpha=alpha,
                eta=eta,
                maxNumIterations=maxNumIterations,
                epsilon=epsilon,
                errors=errors, seed_value=seed_value, numHiddenNodes=numH, letter_list=letter_list,
                outputArrayTracker=outputArrayTracker)

sse = SSETracker.pop(len(SSETracker))
wWeightArray = wWeightTracker.pop(len(wWeightTracker))
vWeightArray = vWeightTracker.pop(len(vWeightTracker))
biasHiddenWeightArray = hiddenBiasTracker.pop(len(hiddenBiasTracker))
biasOutputWeightArray = outputBiasTracker.pop(len(outputBiasTracker))
arraySizeList = obtainNeuralNetworkSizeSpecs(numHiddenNodes=numH)
hiddenArray = []
classes = []
letters = []
lettersSSE = []
classesSSE = []
arraySizeList = obtainNeuralNetworkSizeSpecs(numHiddenNodes=numH, numOutputNodes=9)
for l in range(26):
    x = obtainSelectedAlphabetTrainingValues(l)
    inputDataList = x[1]
    letters.append(x[3])
    classes.append(x[5])
    h = ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList, wWeightArray, biasHiddenWeightArray)
    hiddenArray.append(h)
    o = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, h,
                                               vWeightArray, biasOutputWeightArray)
    errorArray = np.zeros(9)

    desiredOutputArray = np.zeros(9)  # initialize the output array with 0's
    desiredClass = x[4]  # identify the desired class
    desiredOutputArray[desiredClass-1] = 1  # set the desired output for that class to 1

    # Determine the error between actual and desired outputs
    newSSE = 0.0
    for node in range(9):  # Number of nodes in output set (classes)
        errorArray[node] = desiredOutputArray[node] - o[node]
        newSSE = newSSE + errorArray[node] * errorArray[node]

    letterSSE.append(newSSE)

hiddenActivation_df = pd.DataFrame(hiddenArray).transpose()
hiddenActivation_df.columns = letters
column_orders = pd.DataFrame({'A':[obtainSelectedAlphabetTrainingValues(x)[3] for x in range(26)],'Order':[obtainSelectedAlphabetTrainingValues(x)[5] for x in range(26)]})
hiddenActivation_df = hiddenActivation_df.reindex_axis(column_orders.sort_values('Order').A.tolist(),axis=1)
hiddenActivation_df
column_orders = column_orders.sort_values('Order')

plt.figure()
plt.imshow(hiddenActivation_df)
plt.xticks(range(26), column_orders.A+"/"+column_orders.Order)
plt.colorbar()
plt.ylabel('Hidden Node Activation')
plt.xlabel('Letter/Class')
plt.title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \nIter/maxIter: %d/%d  seed: %d  hidden: %d' % (
    alpha, eta, np.mean(errors), epsilon, len(outputArrayTracker), maxNumIterations, seed_value, numH))

