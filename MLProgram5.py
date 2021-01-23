#5. Write a program to implement the naïve Bayesian classifier for a sample training data
#set stored as a .CSV file. Compute the accuracy of the classifier, considering few test
#data sets

print("\nNaive Bayes Classifier for concept learning problem")
import csv
import math
def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def loadCsv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset
 
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    i=0
    while len(trainSet) < trainSize:  # Pop instances from copy to trainset
        trainSet.append(copy.pop(i))  # Reamaning instances in copy are testset
    return [trainSet, copy]
 
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):  #separate dataset as separted={1.0:{Rows where class is 1},0.0:{rows where class is 0}}
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    print("Separated 1.0 and 0.0 class instances")
    print(separated)
    return separated
 
def mean(numbers):
    return safe_div(sum(numbers),float(len(numbers)))

def stdev(numbers):
    avg = mean(numbers)
    variance = safe_div(sum([pow(x-avg,2) for x in numbers]),float(len(numbers)-1))
    return math.sqrt(variance)
 
def summarize(dataset):
    print("ZIP")
    for attr in zip(*dataset):
        print(attr)
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    print("Summaries of mean and stddev")
    print(summaries)
    del summaries[-1]        # Delete the summaries of last column=target
    return summaries
 
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():      # Get class values 1.0 and 0.0
        print(classValue)
        print("Instances")
        print(instances)
        summaries[classValue] = summarize(instances)
    return summaries
 
def calculateProbability(x, mean, stdev):  # Probability of test instance
    exponent = math.exp(-safe_div(math.pow(x-mean,2),(2*math.pow(stdev,2))))
    final = safe_div(1 , (math.sqrt(2*math.pi) * stdev)) * exponent
    print("Attribute Probability")
    print(final)
    return final
 
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items(): #For classvalue as 1.0 and 0.0 
        print(f"Classvalue={classValue}")
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):   #Calculate probabilities of every attribute of testset 
            mean, stdev = classSummaries[i]	   # Get into mean and stddev from summaries of trainset
            print("classsummaries")
            print(classSummaries[i])
            x = inputVector[i]             #Get attribute by attribute value of every 
            print("X")                         #instance of testset 
            print(x)                       
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
        print("Probability of instance")
        print(probabilities[classValue])
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    print(f"Final All Probabilities:{probabilities}")
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            print("Comparison Best Probability class Label")
            bestProb = probability
            bestLabel = classValue
            print(f"predicted classvalue{classValue}")
            print(f"predicted probability{probability}")
          
    return bestLabel
 
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):  #For every instance in testset predict mean and stddev
        print("Testset Instance")
        print(testSet[i])
        result = predict(summaries, testSet[i])
        print("Result")
        print(result)
        predictions.append(result)
    return predictions
 
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    accuracy = safe_div(correct,float(len(testSet))) * 100.0
    return accuracy
 

def main():
    filename = '5ConceptLearning.csv'
    splitRatio = 0.80
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into'.format(len(dataset)))
    print('Number of Training data: ' + (repr(len(trainingSet))))
    print('Number of Test Data: ' + (repr(len(testSet))))
    print("\nThe Training set are:")
    for x in trainingSet:
        print(x)
    print("\nThe Test data set are:")
    for x in testSet:
        print(x)
    # prepare model
    summaries = summarizeByClass(trainingSet)
    print("Final Summaries(mean and stddev) of 1.0 and 0.0 ")
    print(summaries)
    # test model
    predictions = getPredictions(summaries, testSet)
    actual = []
    for i in range(len(testSet)):
        vector = testSet[i]
        actual.append(vector[-1])
    # Since there are five attribute values, each attribute constitutes to 20% accuracy. So if all attributes match with predictions then 100% accuracy
        print('Actual values: {0}%'.format(actual))
        print('Predictions: {0}%'.format(predictions))
        accuracy = getAccuracy(testSet, predictions)
        print('Accuracy: {0}%'.format(accuracy))
 
main()
