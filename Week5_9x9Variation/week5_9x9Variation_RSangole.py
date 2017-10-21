# This code solves the NN for the response variable being the 'classes' of the alphabets
# In total there are 9 classes, thus number of output nodes is 9
# The random weights are initiated in [-0.3, 0.3]
# Sigmoid activation function is used

from math import exp
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

# tinedir = os.path.expanduser('~') + '/Documents/OneDrive/MSPA/490/tineGithub'
# sys.path.insert(0, tinedir)
# from letter import *

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def welcome():
    print 'This program learns to distinguish between many capital letters...'
    return ()

class Letter(object):
    """A class to hold one instance of a given letter.
    """
    def __init__(self, flattened_letter_array, target_letter):
        self.array = flattened_letter_array
        self.letter = target_letter
        self.target = ord(self.letter.lower()) - 97

def load_letters(size=25):
    """ Load all letters of given size (25, 81, etc)

    :param size:
    :return: number of unique letters and a list of Letter objects
    """
    from os import listdir
    basedir = '/Users/Rahul/Documents/OneDrive/MSPA/490/tineGithub/letter'  # set this to match your environment!

    if size == 25:
        alpha_dir = basedir + '/5x5_alphabet/'
    elif size == 81:
        alpha_dir = basedir + '/9x9_alphabet/'

    letters = [f for f in listdir(alpha_dir) if f[0] != '.']
    letter_objects = []

    for letter in letters:
        letter_dir = alpha_dir + letter + '/'
        for instance in [f for f in listdir(letter_dir)]:
            instance_array = []
            with open(letter_dir + instance) as f:
                for line in f:
                    if '#' not in line:
                        instance_array.extend(line.replace(' ','').replace(',','').replace('\n',''))
            letter_objects.append(Letter([int(x) for x in instance_array],letter))

    return len(letters), letter_objects

def split_letters_into_train_and_test(letters, pct_test=20):
    """Split a given set of letters into training and test sets.

    :param letters:
    :param pct_test: a rough percentage of how many letters to hold back for testing
    :return: train, test
    """
    pass

def get_one_example_per_letter(all_letters):
    letters = set([l.letter for l in all_letters])
    letter_objects = []
    while len(letters) > 0:
        let = random.choice(all_letters)
        if let.letter in letters:
            letter_objects.append(let)
            letters.remove(let.letter)
    return letter_objects

# Distortion code from Scott Anderson
def add_distortion(array_to_distort, percent):
    row = 0
    for letter_arr in array_to_distort:
        col = 0
        for letter_item in letter_arr:
            rand_val = random()
            if rand_val <= percent:
                cur_val = array_to_distort[row, col]
                if cur_val == 0:
                    array_to_distort[row, col] = 1
                else:
                    array_to_distort[row, col] = 0

            col += 1

        row += 1
    return (array_to_distort)

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
def obtainNeuralNetworkSizeSpecs(numInputNodes=81, numHiddenNodes=6, numOutputNodes=26):
    return numInputNodes, numHiddenNodes, numOutputNodes

# Single weight randomly picked between -1 and 1
def InitializeWeight():
    # randomNum = random.random()
    # weight = 1 - 2 * randomNum      #returns a random number between -1 and 1
    weight= random.uniform(-0.3, 0.3)
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
def obtainSelectedAlphabetTrainingValues (dataSet):


    trainingDataListA0 =  (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],0,'A',1,'A') # training data list 1 selected for the letter 'A'
    trainingDataListB0 =  (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],1,'B',2,'B') # training data list 2, letter 'E', courtesy AJM
    trainingDataListC0 =  (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],2,'C',3,'C') # training data list 3, letter 'C', courtesy PKVR
    trainingDataListD0 =  (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],3,'D',4,'O') # training data list 4, letter 'D', courtesy TD
    trainingDataListE0 =  (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],4,'E',5,'E') # training data list 5, letter 'E', courtesy BMcD
    trainingDataListF0 =  (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],5,'F',5,'E') # training data list 6, letter 'F', courtesy SK
    trainingDataListG0 =  (7,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],6,'G',3,'C')
    trainingDataListH0 =  (8,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],7,'H',1,'A') # training data list 8, letter 'H', courtesy JC
    trainingDataListI0 =  (9,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],8,'I',6,'I') # training data list 9, letter 'I', courtesy GR
    trainingDataListJ0 = (10,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],9,'J',6,'I') # training data list 10 selected for the letter 'L', courtesy JT
    trainingDataListK0 = (11,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],10,'K',7,'K') # training data list 11 selected for the letter 'K', courtesy EO
    trainingDataListL0 = (12,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],11,'L',8,'L') # training data list 12 selected for the letter 'L', courtesy PV
    trainingDataListM0 = (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],12,'M',9,'M') # training data list 13 selected for the letter 'M', courtesy GR
    trainingDataListN0 = (14,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],13,'N',9,'M') # training data list 14 selected for the letter 'N'
    trainingDataListO0 = (15,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],14,'O',4,'O') # training data list 15, letter 'O', courtesy TD
    trainingDataListP0 = (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],15,'P',2, 'B') # training data list 16, letter 'P', courtesy MT
    trainingDataListQ0 = (17,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],16,'Q',3,'O') # training data list 17, letter 'Q', courtesy AJM (square corners)
    trainingDataListR0 = (18,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],17,'R',1,'B') # training data list 18, letter 'R', courtesy AJM (variant on 'P')
    trainingDataListS0 = (19,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],18,'S',5,'E') # training data list 19, letter 'S', courtesy RG (square corners)
    trainingDataListT0 = (20,[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],19,'T',6,'I') # training data list 20, letter 'T', courtesy JR
    trainingDataListU0 = (21,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,1,1,1,1,0,0],20,'U',8,'L') # training data list 21, letter 'U', courtesy AJM
    trainingDataListV0 = (22,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0],21,'V',8,'I') # training data list 22, letter 'V', courtesy AJM
    trainingDataListW0 = (23,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,1,0,1,0,0,1, 1,0,1,0,0,0,1,0,1, 0,1,0,0,0,0,0,1,0],22,'W',9,'M') # training data list 23, letter 'W', courtesy KW
    trainingDataListX0 = (24,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],23,'X',8,'L') # training data list 24, letter 'X', courtesy JD
    trainingDataListY0 = (25,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],24,'Y',8,'L') # training data list 26, letter 'Z', courtesy ZC
    trainingDataListZ0 = (26,[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],25,'Z', 5,'E') # training data list 26, letter 'Z', courtesy ZW

    if dataSet == 0: trainingDataList = trainingDataListA0
    if dataSet == 1: trainingDataList = trainingDataListB0
    if dataSet == 2: trainingDataList = trainingDataListC0
    if dataSet == 3: trainingDataList = trainingDataListD0
    if dataSet == 4: trainingDataList = trainingDataListE0
    if dataSet == 5: trainingDataList = trainingDataListF0
    if dataSet == 6: trainingDataList = trainingDataListG0
    if dataSet == 7: trainingDataList = trainingDataListH0
    if dataSet == 8: trainingDataList = trainingDataListI0
    if dataSet == 9: trainingDataList = trainingDataListJ0
    if dataSet == 10: trainingDataList = trainingDataListK0
    if dataSet == 11: trainingDataList = trainingDataListL0
    if dataSet == 12: trainingDataList = trainingDataListM0
    if dataSet == 13: trainingDataList = trainingDataListN0
    if dataSet == 14: trainingDataList = trainingDataListO0
    if dataSet == 15: trainingDataList = trainingDataListP0
    if dataSet == 16: trainingDataList = trainingDataListQ0
    if dataSet == 17: trainingDataList = trainingDataListR0
    if dataSet == 18: trainingDataList = trainingDataListS0
    if dataSet == 19: trainingDataList = trainingDataListT0
    if dataSet == 20: trainingDataList = trainingDataListU0
    if dataSet == 21: trainingDataList = trainingDataListV0
    if dataSet == 22: trainingDataList = trainingDataListW0
    if dataSet == 23: trainingDataList = trainingDataListX0
    if dataSet == 24: trainingDataList = trainingDataListY0
    if dataSet == 25: trainingDataList = trainingDataListZ0
    # if dataSet == 26:

    return (trainingDataList)

# Gives a random letter. Can specify how many letters to choose from.
def obtainRandomAlphabetTrainingValues (numTrainingDataSets):
    # The training data list will have the  values for the X-OR problem:
    #   - First 81 values will be the 9x9 pixel-grid representation of the letter
    #       represented as a 1-D array (0 or 1 for each)
    #   - 82nd value will be the output class (0 .. totalClasses - 1)
    #   - 83rd value will be the string associated with that class, e.g., 'X'
    # We are starting with five letters in the training set: X, M, N, H, and A
    # Thus there are five choices for training data, which we'll select on random basis

    dataSet = random.randint(0, numTrainingDataSets)

    trainingDataListA0 =  (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],0,'A',1,'A') # training data list 1 selected for the letter 'A'
    trainingDataListB0 =  (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],1,'B',2,'B') # training data list 2, letter 'E', courtesy AJM
    trainingDataListC0 =  (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],2,'C',3,'C') # training data list 3, letter 'C', courtesy PKVR
    trainingDataListD0 =  (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],3,'D',4,'O') # training data list 4, letter 'D', courtesy TD
    trainingDataListE0 =  (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],4,'E',5,'E') # training data list 5, letter 'E', courtesy BMcD
    trainingDataListF0 =  (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],5,'F',5,'E') # training data list 6, letter 'F', courtesy SK
    trainingDataListG0 =  (7,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],6,'G',3,'C')
    trainingDataListH0 =  (8,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],7,'H',1,'A') # training data list 8, letter 'H', courtesy JC
    trainingDataListI0 =  (9,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],8,'I',6,'I') # training data list 9, letter 'I', courtesy GR
    trainingDataListJ0 = (10,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],9,'J',6,'I') # training data list 10 selected for the letter 'L', courtesy JT
    trainingDataListK0 = (11,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],10,'K',7,'K') # training data list 11 selected for the letter 'K', courtesy EO
    trainingDataListL0 = (12,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],11,'L',8,'L') # training data list 12 selected for the letter 'L', courtesy PV
    trainingDataListM0 = (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],12,'M',9,'M') # training data list 13 selected for the letter 'M', courtesy GR
    trainingDataListN0 = (14,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],13,'N',9,'M') # training data list 14 selected for the letter 'N'
    trainingDataListO0 = (15,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],14,'O',4,'O') # training data list 15, letter 'O', courtesy TD
    trainingDataListP0 = (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],15,'P',2, 'B') # training data list 16, letter 'P', courtesy MT
    trainingDataListQ0 = (17,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],16,'Q',3,'O') # training data list 17, letter 'Q', courtesy AJM (square corners)
    trainingDataListR0 = (18,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],17,'R',1,'B') # training data list 18, letter 'R', courtesy AJM (variant on 'P')
    trainingDataListS0 = (19,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],18,'S',5,'E') # training data list 19, letter 'S', courtesy RG (square corners)
    trainingDataListT0 = (20,[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],19,'T',6,'I') # training data list 20, letter 'T', courtesy JR
    trainingDataListU0 = (21,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,1,1,1,1,0,0],20,'U',8,'L') # training data list 21, letter 'U', courtesy AJM
    trainingDataListV0 = (22,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0],21,'V',8,'I') # training data list 22, letter 'V', courtesy AJM
    trainingDataListW0 = (23,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,1,0,1,0,0,1, 1,0,1,0,0,0,1,0,1, 0,1,0,0,0,0,0,1,0],22,'W',9,'M') # training data list 23, letter 'W', courtesy KW
    trainingDataListX0 = (24,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],23,'X',8,'L') # training data list 24, letter 'X', courtesy JD
    trainingDataListY0 = (25,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],24,'Y',8,'L') # training data list 26, letter 'Z', courtesy ZC
    trainingDataListZ0 = (26,[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],25,'Z', 5,'E') # training data list 26, letter 'Z', courtesy ZW


    if dataSet == 0: trainingDataList = trainingDataListA0
    if dataSet == 1: trainingDataList = trainingDataListB0
    if dataSet == 2: trainingDataList = trainingDataListC0
    if dataSet == 3: trainingDataList = trainingDataListD0
    if dataSet == 4: trainingDataList = trainingDataListE0
    if dataSet == 5: trainingDataList = trainingDataListF0
    if dataSet == 6: trainingDataList = trainingDataListG0
    if dataSet == 7: trainingDataList = trainingDataListH0
    if dataSet == 8: trainingDataList = trainingDataListI0
    if dataSet == 9: trainingDataList = trainingDataListJ0
    if dataSet == 10: trainingDataList = trainingDataListK0
    if dataSet == 11: trainingDataList = trainingDataListL0
    if dataSet == 12: trainingDataList = trainingDataListM0
    if dataSet == 13: trainingDataList = trainingDataListN0
    if dataSet == 14: trainingDataList = trainingDataListO0
    if dataSet == 15: trainingDataList = trainingDataListP0
    if dataSet == 16: trainingDataList = trainingDataListQ0
    if dataSet == 17: trainingDataList = trainingDataListR0
    if dataSet == 18: trainingDataList = trainingDataListS0
    if dataSet == 19: trainingDataList = trainingDataListT0
    if dataSet == 20: trainingDataList = trainingDataListU0
    if dataSet == 21: trainingDataList = trainingDataListV0
    if dataSet == 22: trainingDataList = trainingDataListW0
    if dataSet == 23: trainingDataList = trainingDataListX0
    if dataSet == 24: trainingDataList = trainingDataListY0
    if dataSet == 25: trainingDataList = trainingDataListZ0

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
                                        biasHiddenWeightArray, vWeightArray, biasOutputWeightArray, verbose = False):
    errors = []
    selectedTrainingDataSet = 0
    # inputArrayLength = arraySizeList[0]
    # hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    while selectedTrainingDataSet < numTrainingDataSets:
        trainingDataList = obtainSelectedAlphabetTrainingValues(dataSet=selectedTrainingDataSet)
        inputDataList = trainingDataList[1]

        # for node in range(inputArrayLength):
        #     trainingData = trainingDataInputList[node]
        #     inputDataList.append(trainingData)

        if verbose:
            print '   >Data Set Number', selectedTrainingDataSet, 'for letter ', trainingDataList[3]

        hiddenArray = ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList, wWeightArray,
                                                            biasHiddenWeightArray)

        # print '   >The hidden node activations are:'
        # print hiddenArray

        outputArray = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hiddenArray, vWeightArray,
                                                             biasOutputWeightArray)

        if verbose:
            print '   >The output node activations are:', outputArray

        desiredOutputArray = np.zeros(outputArrayLength)  # initialize the output array with 0's
        desiredClass = trainingDataList[4]  # identify the desired class
        desiredOutputArray[desiredClass-1] = 1  # set the desired output for that class to 1

        if verbose:
            print '   >The desired output array values are: ', desiredOutputArray

        # Determine the error between actual and desired outputs

        # Initialize the error array
        errorArray = np.zeros(outputArrayLength)

        newSSE = 0.0
        for node in range(outputArrayLength):  # Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node] * errorArray[node]

        if verbose:
            print '   >The error values are:', errorArray

        if verbose:
            # Print the Summed Squared Error
            print '   >New SSE = %.6f' % newSSE

        errors.append(newSSE)

        selectedTrainingDataSet = selectedTrainingDataSet + 1

    return errors

# Check if errors for all letters are below Epsilon
def allErrorsBelowEpsilon(errors, epsilon):
    for error in errors:
        if error > epsilon:
            return False

    # got here and no errors > epsilon
    return True

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
def main(alpha = 1.0, eta = 0.5, maxNumIterations = 5000, epsilon = 0.05, numTrainingDataSets = 25, seed_value = 1, numHiddenNodes = 6):
    random.seed(seed_value)

    welcome()

    iteration = 0

    print '1. Setting up network...'
    # Obtain the actual sizes for each layer of the network
    arraySizeList = obtainNeuralNetworkSizeSpecs(numHiddenNodes = numHiddenNodes, numOutputNodes = 9)

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
    ComputeOutputsAcrossAllTrainingData(alpha, arraySizeList, numTrainingDataSets, wWeightArray, biasHiddenWeightArray,
                                        vWeightArray, biasOutputWeightArray)
    print '....Done'

    print '4. Initiating trackers...'
    # Variables to track changes in wWeight, vWeight, hiddenBias, outputBias, and SSE
    wWeightTracker = dict()
    vWeightTracker = dict()
    hiddenBiasTracker = dict()
    outputBiasTracker = dict()
    SSETracker = dict()
    letterTracker = dict()
    classTracker = dict()
    outputArrayTracker = dict()
    print '....Done'

    # Next step - Obtain a single set of randomly-selected training values for alpha-classification
    print '5. Starting convergence iterations...'
    while iteration < maxNumIterations:

        # Increment the iteration count
        iteration = iteration + 1

        # For any given pass, we re-initialize the training list
        inputDataList = []

        # Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
        trainingDataList = obtainRandomAlphabetTrainingValues(numTrainingDataSets)

        inputDataList = trainingDataList[1]

        desiredOutputArray = np.zeros(outputArrayLength)  # initialize the output array with 0's
        desiredClass = trainingDataList[4]  # identify the desired class
        desiredOutputArray[desiredClass-1] = 1  # set the desired output for that class to 1

        # Compute a single feed-forward pass and obtain the Actual Outputs
        hiddenArray = ComputeSingleFeedforwardPassFirstStep(alpha, arraySizeList, inputDataList,
                                                        wWeightArray, biasHiddenWeightArray)

        outputArray = ComputeSingleFeedforwardPassSecondStep(alpha, arraySizeList, hiddenArray,
                                                             vWeightArray, biasOutputWeightArray)

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

        # Perform second part of the backpropagation of weight changes
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

        print '{*} For iteration %d, letter %s, SSE = %.6f' %(iteration-1, trainingDataList[3], newSSE)

        # Tracker functionality
        wWeightTracker[iteration] = wWeightArray
        vWeightTracker[iteration] = vWeightArray
        hiddenBiasTracker[iteration] = biasHiddenWeightArray
        outputBiasTracker[iteration] = biasOutputWeightArray
        SSETracker[iteration] = newSSE
        letterTracker[iteration] = trainingDataList[3]
        classTracker[iteration] = trainingDataList[5]
        outputArrayTracker[iteration] = outputArray

        if newSSE < epsilon:
            errors = ComputeOutputsAcrossAllTrainingData(alpha, arraySizeList, numTrainingDataSets, wWeightArray,
                                                         biasHiddenWeightArray, vWeightArray, biasOutputWeightArray)
            if allErrorsBelowEpsilon(errors, epsilon):
                break


    print '\n** Out of while loop at iteration **', iteration

    # After training, get a new comparative set of outputs, errors, and SSE
    print ' '
    print '  After training:'
    ComputeOutputsAcrossAllTrainingData(alpha, arraySizeList, numTrainingDataSets, wWeightArray, biasHiddenWeightArray, vWeightArray, biasOutputWeightArray)

    return vWeightTracker, wWeightTracker, hiddenBiasTracker, outputBiasTracker, SSETracker, letterTracker, outputArrayTracker, errors, classTracker


def plotSSE(SSETracker, letterTracker, alpha, eta,
            maxNumIterations, epsilon, errors, numTrainingDataSets,
            seed_value,numHiddenNodes=6):
    x, y = zip(*SSETracker.items())
    x, letters = zip(*letterTracker.items())
    df = pd.DataFrame(dict(iterations=x, SSE=y, letters=letters))
    groups = df.groupby('letters')
    # sns.lmplot(x='iterations', y='SSE', data=df,
    #            fit_reg=False, hue='letters', lowess=True)

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.iterations, group.SSE, marker='o', linestyle='', ms=4,
                label=name, alpha = .6)
    ax.legend(ncol=2)
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('SSE')
    ax.set_title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \nIter/maxIter: %d/%d  seed: %d  hidden: %d' %(alpha, eta, np.mean(errors), epsilon, df.shape[0], maxNumIterations, seed_value,numHiddenNodes))
    ax.set_ylim([0, 1.5])

    plt.show()

def plotSubplots(SSETracker, letterTracker, alpha, eta,
            maxNumIterations, epsilon, errors, numTrainingDataSets,
            seed_value, axi, numHiddenNodes=6):
    x, y = zip(*SSETracker.items())
    x, letters = zip(*letterTracker.items())
    df = pd.DataFrame(dict(iterations=x, SSE=y, letters=letters))
    groups = df.groupby('letters')
    # sns.lmplot(x='iterations', y='SSE', data=df,
    #            fit_reg=False, hue='letters', lowess=True)

    # Plot
    # axi.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        axi.plot(group.iterations, group.SSE, marker='o', linestyle='', ms=4,
                label=name, alpha = .6)
    axi.legend()
    axi.set_xlabel('Iteration Number')
    axi.set_ylabel('SSE')
    axi.set_title('alpha: %1.1f  eta: %1.1f  \nsse/epsilon: %1.3f/%1.3f  \nIter/maxIter: %d/%d  \nseed: %d  hidden: %d' %(alpha, eta, np.mean(errors), epsilon, df.shape[0], maxNumIterations, seed_value,numHiddenNodes))
    axi.set_ylim([0, 1.5])

def plotLetter(trainingDataList):
    pixelArray = np.reshape(trainingDataList[1], (9, 9))

    fig, ax = plt.subplots()
    im = ax.matshow(pixelArray, cmap='Wistia')
    locs = np.arange(len(pixelArray))
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_ticks(locs + 0.5, minor=True)
        axis.set(ticks=locs, ticklabels=np.arange(0, 9))
    ax.grid(True, which='minor', color='black', linewidth=1)
    ax.set_title('Letter: %s' %trainingDataList[3])
    plt.show()

    return

def plotSSEbyLetter(alpha, eta, maxNumIterations, epsilon, errors, seed_value, numHiddenNodes, letter_list, outputArrayTracker):
    plt.figure()
    plt.bar(range(25), errors)
    plt.hlines(y=epsilon,xmin=0,xmax=25,colors='red')
    plt.xticks(range(25), letter_list)
    plt.ylabel('Letter specific SSE')
    plt.xlabel('Letter')
    plt.title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \niter/maxIter: %d/%d  seed: %d  hidden: %d' % (
        alpha, eta, np.mean(errors), epsilon, len(outputArrayTracker), maxNumIterations, seed_value, numHiddenNodes))

def plotSSEbyClass(SSETracker, classTracker, alpha, eta,
            maxNumIterations, epsilon, errors, numTrainingDataSets,
            seed_value,numHiddenNodes):
    x, y = zip(*SSETracker.items())
    x, classes = zip(*classTracker.items())
    df = pd.DataFrame(dict(iterations=x, SSE=y, classes=classes))
    groups = df.groupby('classes')

    # Plot
    fig, ax = plt.subplots()
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for name, group in groups:
        ax.plot(group.iterations, group.SSE, marker='o', linestyle='', ms=4,
                label=name, alpha = .6)
    ax.legend(ncol=2)
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('SSE')
    ax.set_title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \niter/maxIter: %d/%d  seed: %d  hidden: %d' %(alpha, eta, np.mean(errors), epsilon, df.shape[0], maxNumIterations, seed_value,numHiddenNodes))
    ax.set_ylim([0, 1.5])

    plt.show()


def plotHiddenHeatmap(hiddenArray, classes, alpha, eta,
outputArrayTracker, maxNumIterations, epsilon, errors,
                      seed_value, numHiddenNodes):
    hiddenActivation_df = pd.DataFrame(hiddenArray).transpose()
    hiddenActivation_df.columns = classes
    hiddenActivation_df
    plt.figure()
    plt.imshow(hiddenActivation_df)
    plt.xticks(range(9), classes)
    plt.colorbar()
    plt.ylabel('Hidden Node Activation')
    plt.xlabel('Letter')
    plt.title('alpha: %1.1f  eta: %1.1f  sse/epsilon: %1.3f/%1.3f  \niter/maxIter: %d/%d  seed: %d  hidden: %d' % (
        alpha, eta, np.mean(errors), epsilon, len(outputArrayTracker), maxNumIterations, seed_value, numHiddenNodes))

