from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)
    
def stddev(values):
    mean = sum(values)/float(len(values))
    differencetotal = 0
    for value in values:
        differencetotal += abs(value - mean)**2
    differencetotal /= float(len(values))
    return sqrt(differencetotal)

#testCarData()

# def q5():
#     print('q5')
#     pen = []
#     car = []
#     for i in range(5):
#         pen.append(testPenData()[1])
#         print("%d times pen DONE" % (i + 1))
#         car.append(testCarData()[1])
#         print("%d times car DONE" % (i + 1))
# 
# 
#     print('Pen data set:')
#     print("Max %s | Avg %s | STD %s" % (max(pen), average(pen), stDeviation(pen)))
#     print('Car data set:')
#     print("Max %s | Avg %s | STD %s" % (max(car), average(car), stDeviation(car)))
# 
# q5()

# 
# def q6():
#     print ('q6')
#     print("******************************************")
#     for y in range(0, 41, 5):
#         pen = []
#         car = []
#         for x in range(5):
#             print("Pen test number:", x + 1, "y:", y)
#             result = testPenData([y])
#             pen.append(result[1])
#             print ("Car test number:", x + 1, "y:", y)
#             result = testCarData([y])
#             car.append(result[1])
#         print("Pen Data Avg, Max, Stdev: " + str((sum(pen) / float(len(pen)))) + ", " + str(max(pen)) + ", " + str(
#             stddev(pen)) + "\n")
#         print("Car Data Avg, Max, Stdev: " + str((sum(car) / float(len(car)))) + ", " + str(max(car)) + ", " + str(
#             stddev(car)) + "\n")
# 
# q6()
#print(type(penData[1]))
truthTable = [([0, 0], [0]),
			  ([1, 0], [1]),
			  ([0, 1], [1]),
			  ([1, 1], [0])]

def buildNet(hiddenLayer=[]):
	accuracies = []
	for i in range(5):
		nnet, accuracy = buildNeuralNet(examples=(truthTable, truthTable), weightChangeThreshold = 0.0001, hiddenLayerList=hiddenLayer, maxItr= 2000)
		accuracies.append(accuracy)
	return float(sum(accuracies)) / float(len(accuracies))


results=[]

results.append(buildNet([]))
'''
while results[-1] != 1.0:
	results.append(buildNet([i]))
	i+= 1
'''
for i in range(1,51):
    results.append(buildNet([i]))
    
for i in range(len(results)):
	print ("%d\t%f"%(i, results[i]))
	print(results[i])