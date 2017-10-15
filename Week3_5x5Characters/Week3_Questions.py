# Question 1:
# Get a convergent solution for all letters:
alpha = 1.0
eta = 0.5
maxNumIterations = 10000
epsilon = 0.1
numTrainingDataSets = 4
seed_value = 1

vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
    alpha=alpha,
    eta=eta,
    maxNumIterations=maxNumIterations,
    epsilon=epsilon,
    numTrainingDataSets=numTrainingDataSets,
    seed_value=seed_value,
    numHiddenNodes = 10
)
plotSSE(SSETracker, letterTracker, alpha=alpha,
    eta=eta,
    maxNumIterations=maxNumIterations,
    epsilon=epsilon,
    numTrainingDataSets=numTrainingDataSets,
    seed_value=seed_value)

# Question 2:
# Explore sensitivity of SSE to alpha
eta = 0.5
maxNumIterations = 1000
epsilon = 0.1
numTrainingDataSets = 4
seed_value = 1
maxNumIterations = 10000
for a in [0.5,1.0,1.5]:
    vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
        alpha=a,
        eta=eta,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value
    )
    plotSSE(SSETracker, letterTracker, alpha=a,
            eta=eta,
            maxNumIterations=maxNumIterations,
            epsilon=epsilon,
            numTrainingDataSets=numTrainingDataSets,
            seed_value=seed_value)

# Explore sensitivity of SSE to eta
alpha = 1.5
maxNumIterations = 10000
epsilon = 0.1
numTrainingDataSets = 4
seed_value = 1
numH = 4
for e in [1,1.5,2.0]:
    vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
        alpha=alpha,
        eta=e,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value,
        numHiddenNodes = numH
    )
    plotSSE(SSETracker, letterTracker, alpha=alpha,
            eta=e,
            maxNumIterations=maxNumIterations,
            epsilon=epsilon,
            numTrainingDataSets=numTrainingDataSets,
            seed_value=seed_value)

# Both together now
maxNumIterations = 3000
epsilon = 0.1
numTrainingDataSets = 4
seed_value = 1
alpha = np.arange(0.4,2.1,.1)
eta   = np.arange(0.1,2.1,.1)
tuneGrid = np.zeros((len(alpha),len(eta)))
for a in range(len(alpha)):
    for e in range(len(eta)):
        vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
            alpha=alpha[a],
            eta=eta[e],
            maxNumIterations=maxNumIterations,
            epsilon=epsilon,
            numTrainingDataSets=numTrainingDataSets,
            seed_value=seed_value
        )
        tuneGrid[a,e]=len(SSETracker)

tuneGrid
plt.figure()
CS = plt.contour(eta, alpha, tuneGrid,linestyles='dashed')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contours of iterations reached before epsilon reaches 0.1 \n (maxIter = 3000)')
plt.xlabel('eta')
plt.ylabel('alpha')


# tuneGrid = np.load('tuneGrid.p')
# np.amin(tuneGrid,axis=1)
# Lowest value of iterations for eta = 2.1, alpha = 1.3

# Question 4
# Explore sensitivity of SSE & convergence iterations to initial weight arrays
alpha = 1.3
eta = 2.1
maxNumIterations = 3000
epsilon = 0.1
numTrainingDataSets = 4
seed_value =np.arange(1,100)
vW = dict()
wW = dict()
hiddenBias = dict()
outputBias = dict()
SSE = dict()
iterStop = dict()
letters = dict()
op = dict()
for s in range(len(seed_value)):
    vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
        alpha=alpha,
        eta=eta,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value[s]
    )
    wW[s] = vWeightTracker.pop(len(vWeightTracker))
    vW[s] = wWeightTracker.pop(len(wWeightTracker))
    hiddenBias[s] = hiddenBiasTracker.pop(len(hiddenBiasTracker))
    outputBias[s] = outputBiasTracker.pop(len(outputBiasTracker))
    iterStop[s] = len(SSETracker)
    SSE[s] = SSETracker.pop(len(SSETracker))
    letters[s] = letterTracker.pop(len(letterTracker))
    op[s] = outputArrayTracker.pop(len(outputArrayTracker))

i, sse = zip(*SSE.items())
plt.hist(sse,bins=30)
plt.title('Histogram of SSE for 100 random starts')
plt.xlabel('SSE')
plt.ylabel('Count')

# w weights ->
d = pd.DataFrame(wW.items(),columns=['seed','value'])

wWaverages = np.zeros([5,6])
wWstd = np.zeros([5,6])
for o in np.arange(5): #5 outputs
    for h in np.arange(6): #6 hidden
        wWaverages[o][h] = np.mean([d.value[i][o][h] for i in np.arange(d.shape[0])])
        wWstd[o][h] = np.std([d.value[i][o][h] for i in np.arange(d.shape[0])])
wWaverages
wWstd

for h in np.arange(5):
    plt.hist([d.value[i][0][h] for i in np.arange(d.shape[0])],alpha = .7)
    plt.title('Weights for 6 hidden nodes in W array for output=0 node over 100 runs')
    plt.xlabel('Weight')

# v weights ->
d = pd.DataFrame(vW.items(),columns=['seed','value'])

vWaverages = np.zeros([6,25])
vWstd = np.zeros([6,25])
for i in np.arange(25): #5 outputs
    for h in np.arange(6): #6 hidden
        vWaverages[h][i] = np.mean([d.value[x][h][i] for x in np.arange(d.shape[0])])
        vWstd[h][i] = np.std([d.value[x][h][i] for x in np.arange(d.shape[0])])
vWaverages
vWstd

for h in np.arange(25):
    plt.hist([d.value[x][h][0] for x in np.arange(d.shape[0])],alpha = .3)
    plt.title('Weights for 25 input nodes in V array for hidden=0 node over 100 runs')
    plt.xlabel('Weight')

plt.hist([iterStop[x] for x in np.arange(len(iterStop))],bins=50)
plt.title('Hist of iterations at which SSE < (eplison=0.1) for 100 starts')
plt.xlabel('Stop Iteration Number')


# Question 3:
# Explore sensitivity of SSE to # of hidden nodes
alpha = 1.3
eta = 2.1
maxNumIterations = 5000
epsilon = 0.1
numTrainingDataSets = 4
seed_value = 1
numH = [2,8,14,20,26]
f, axarr = plt.subplots(1, 5, sharey=True)
for i, row in zip(np.arange(len(numH)), axarr):
    hidden_nodes = numH[i]
    vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
        alpha=alpha,
        eta=eta,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value,
        numHiddenNodes = hidden_nodes
    )
    plotSubplots(SSETracker, letterTracker, alpha=alpha,
            eta=eta,
            maxNumIterations=maxNumIterations,
            epsilon=epsilon,
            numTrainingDataSets=numTrainingDataSets,
            seed_value=seed_value,
            axi = row,
            numHiddenNodes=hidden_nodes)

# Trying a large epsilon, with small iterations
alpha = 1.3
eta = 3
maxNumIterations = 500
epsilon = 0.01
numTrainingDataSets = 4
seed_value = 2
numH = [2,8,14,20,26]
# numH = np.arange(2,24,2)
sse = pd.DataFrame(columns=['iter','sse','numh'])
plotter = True
if plotter:
    f, axarr = plt.subplots(1, 5, sharey=True)
    f, axarr2 = plt.subplots(5,1, sharex=True)
for i, row, row2 in zip(np.arange(len(numH)), axarr, axarr2):
# for i in np.arange(len(numH)):
    hidden_nodes = numH[i]
    vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker = main(
        alpha=alpha,
        eta=eta,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value,
        numHiddenNodes = hidden_nodes
    )
    sse_i = pd.DataFrame(SSETracker.items(), columns=['iter', 'sse'])
    sse_i['numh']=hidden_nodes
    sse=sse.append(sse_i)
    if plotter:
        plotSubplots(SSETracker, letterTracker, alpha=alpha,
                     eta=eta,
                     maxNumIterations=maxNumIterations,
                     epsilon=epsilon,
                     numTrainingDataSets=numTrainingDataSets,
                     seed_value=seed_value,
                     axi = row,
                     numHiddenNodes=hidden_nodes)
        row2.plot(sse_i.iter,sse_i.sse,marker='o',ms=3,label=hidden_nodes,alpha=.6,linestyle ='')
        row2.set_title('Number of hidden nodes: %d' %hidden_nodes)
        row2.margins(0.05)

# sse_grouped= sse.groupby('numh')
# fig, ax = plt.subplots()
# for name, group in sse_grouped:
#     ax.plot(group.iter, group.sse, marker='o', ms=4,
#             label=name, alpha=.3, linestyle ='')
# ax.legend()
# ax.set_xlabel('Iteration Number')
# ax.set_ylabel('SSE')
# ax.set_title('alpha: %1.1f  eta: %1.1f  epsilon: %1.1f  Iter/maxIter: %d/%d  seed: %d  hidden: %d' % (
# alpha, eta, epsilon, df.shape[0], maxNumIterations, seed_value, numHiddenNodes))
# ax.set_ylim([0, 1.5])


# Investigation into the hidden activations
alpha = 1.2
eta = 1.5
maxNumIterations = 20000
epsilon = 0.01
numTrainingDataSets = 4
seed_value = 2440
numH = 3
vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker, errors = main(
        alpha=alpha,
        eta=eta,
        maxNumIterations=maxNumIterations,
        epsilon=epsilon,
        numTrainingDataSets=numTrainingDataSets,
        seed_value=seed_value,
        numHiddenNodes = numH
    )
sse = SSETracker.pop(len(SSETracker))
wWeightArray = wWeightTracker.pop(len(wWeightTracker))
vWeightArray = vWeightTracker.pop(len(vWeightTracker))
biasHiddenWeightArray = hiddenBiasTracker.pop(len(hiddenBiasTracker))
biasOutputWeightArray = outputBiasTracker.pop(len(outputBiasTracker))
arraySizeList = obtainNeuralNetworkSizeSpecs(numHiddenNodes=numH)
hiddenArray = []
letters=[]
letterSSE = []
for l in range(5):
    x=obtainSelectedAlphabetTrainingValues(l)
    inputDataList = x[0:25]
    letters.append(x[26])
    h=ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList, wWeightArray, biasHiddenWeightArray)
    hiddenArray.append(h)
    o=ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, h,
                                                     vWeightArray, biasOutputWeightArray)
    errorArray = np.zeros(5)

    desiredOutputArray = np.zeros(5)  # initialize the output array with 0's
    desiredClass = x[25]  # identify the desired class
    desiredOutputArray[desiredClass] = 1  # set the desired output for that class to 1

    # Determine the error between actual and desired outputs
    newSSE = 0.0
    for node in range(5):  # Number of nodes in output set (classes)
        errorArray[node] = desiredOutputArray[node] - o[node]
        newSSE = newSSE + errorArray[node] * errorArray[node]

    letterSSE.append(newSSE)

hiddenArray = pd.DataFrame(hiddenArray).transpose()
hiddenArray.columns = letters
hiddenArray
plt.figure()
plt.imshow(hiddenArray)
plt.xticks([0,1,2,3,4],letters)
plt.colorbar()
plt.ylabel('Hidden Node Activation')
plt.xlabel('Letter')
plt.title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \nIter/maxIter: %d/%d  seed: %d  hidden: %d' % (
alpha, eta, np.mean(errors), epsilon, len(outputArrayTracker), maxNumIterations, seed_value, numH))

plt.figure()
plt.bar([0,1,2,3,4],letterSSE)
plt.xticks([0,1,2,3,4],letters)
plt.ylabel('Letter specific SSE')
plt.xlabel('Letter')
plt.title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \nIter/maxIter: %d/%d  seed: %d  hidden: %d' % (
alpha, eta, np.mean(errors), epsilon, len(outputArrayTracker), maxNumIterations, seed_value, numH))
