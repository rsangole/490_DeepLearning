# -*- coding: utf-8 -*-
import csv
import numpy as np

####################################################################################################
#**************************************************************************************************#
#################################################################################################### 

def createWeightArray ():
# Initialize the weight variables with random weights    
    numUpperNodes = hiddenArrayLength
    numLowerNodes = inputArrayLength
    
    weightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's
    weightVal = 0
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            weightVal = weightVal + 1
            weightArray[row,col] = weightVal
    print ' '
    print ' In createWeightArray'
    print ' The newly created weight matrix is:'
    print weightArray
    return weightArray

####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def convertArrayToList (weightArray): 
    weightList = list()
    numUpperNodes = hiddenArrayLength
    numLowerNodes = inputArrayLength
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            localWeight = weightArray[row,col] 
            weightList.append(localWeight)      

    print ' '
    print ' Transforming the weight matrix to a list yields:'
    print weightList

    return weightList

####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def writeWeightFile (GB1WeightList): 

    GB1WeightFile = open('Canopy/datafiles/GB1WeightFile', 'w') 

    for item in GB1WeightList:
        GB1WeightFile.write("%s\n" % item)  
# Note: This writes the weights to a list, one weight per line, so the subsequent readfile procedure
#   can be modified to read in one data element from each line
    
    GB1WeightFile.close()     

    print ' '
    print ' The weight list has been stored to the file Canopy/data/GB1Weightfile'                        
    return


####################################################################################################
#**************************************************************************************************#
####################################################################################################    

def readWeightFile (): 

    weightList = list()
    with open('Canopy/datafiles/GB1WeightFile', "r") as infile:

        reader = csv.reader(infile)
        for row in reader:
#        print row
            colnum = 0
            theRow = row
            for col in row:
#                print theRow[colnum], col
                data = float(theRow[colnum])
#                print data
            weightList.append(data)
    print ' '
    print ' Reading the weights back from the file:'
    print weightList        
    return weightList                                                  

####################################################################################################
#**************************************************************************************************#
####################################################################################################

def reconstructWeightArray (GB1WeightList):

    numUpperNodes = hiddenArrayLength
    numLowerNodes = inputArrayLength 
    
    GB1WeightArray = np.zeros((numUpperNodes,numLowerNodes))    # iniitalize the weight matrix with 0's     
  
    for row in range(numUpperNodes):  #  Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input matrix (expressed as a column)
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes.
        for col in range(numLowerNodes):  # number of columns in matrix 2
            localPosition = row*numLowerNodes + col
            localWeight = GB1WeightList[localPosition]
            GB1WeightArray[row,col] = localWeight
    print ' '
    print ' In reconstructWeightArray'  
    print ' The recovered weight matrix is: '
    print GB1WeightArray
                                                     
    return GB1WeightArray  


####################################################################################################
#**************************************************************************************************#
####################################################################################################


def main():

# Define the global variables        
    global inputArrayLength
    global hiddenArrayLength
    global outputArrayLength
    
    inputArrayLength = 3
    hiddenArrayLength = 2
    outputArrayLength = 0
    
# Create the initial weight array
    weightArray = createWeightArray ()

# Transform the initial weight array into a list  
    weightList = convertArrayToList (weightArray)

# Write the list into a weights file   
    writeWeightFile (weightList)
    
# Read the weights back into the program, into a list; return the list
    GB1WeightList = readWeightFile()
    
# Convert the weight list back into a weight array
    GB1WeightArray = reconstructWeightArray (GB1WeightList) 
    
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()

####################################################################################################
# End program
#################################################################################################### 

    
             