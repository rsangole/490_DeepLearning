## RAHUL SANGOLE's SUBMISSIONS:

# Plotting function which could help future classes
def plotLetter(trainingDataList):
    pixelArray = np.reshape(trainingDataList[1], (9, 9))

    fig, ax = plt.subplots()
    im = ax.matshow(pixelArray, cmap='Wistia')
    locs = np.arange(len(c))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs, ticklabels=np.arange(0, 9))
    ax.grid(True, which='minor', color='black', linewidth=1)
    ax.set_title('Letter: %s' %trainingDataList[3])
    plt.show()

    return

# My variants of the letter 'R'

trainingDataListR1 = (101,
                      [1, 1, 1, 1, 1, 1, 1, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 0,
                       1, 0, 0, 0, 0, 1, 0, 0, 0,
                       1, 0, 0, 0, 0, 0, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 0, 1],
                      18,
                      'R')

trainingDataListR2 = (102,
                      [1, 1, 1, 1, 1, 1, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 1, 1, 1, 1, 1, 1, 0, 0,
                       1, 0, 0, 0, 0, 1, 0, 0, 0,
                       1, 0, 0, 0, 0, 0, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0],
                      18,
                      'R')

trainingDataListR3 = (103,
                      [1, 1, 1, 1, 1, 1, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 1, 1, 1, 1, 1, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 1, 0, 0,
                       1, 0, 0, 0, 0, 0, 0, 1, 0,
                       1, 0, 0, 0, 0, 0, 0, 0, 1],
                      18,
                      'R')

trainingDataListR4 = (104,
                      [0, 1, 1, 1, 1, 1, 1, 0, 0,
                       0, 1, 0, 0, 0, 0, 0, 1, 0,
                       0, 1, 0, 0, 0, 0, 0, 0, 1,
                       0, 1, 0, 0, 0, 0, 0, 0, 1,
                       0, 1, 0, 0, 0, 0, 0, 1, 0,
                       0, 1, 1, 1, 1, 1, 1, 0, 0,
                       0, 1, 0, 0, 0, 0, 1, 0, 0,
                       0, 1, 0, 0, 0, 0, 0, 1, 0,
                       0, 1, 0, 0, 0, 0, 0, 0, 1],
                      18,
                      'R')