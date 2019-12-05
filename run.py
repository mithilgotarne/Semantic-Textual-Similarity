import sys, nltk
from CorpusReader import CorpusReader
import Input
from sklearn import svm
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    filepath1 = sys.argv[1]
    filepath2 = sys.argv[2]
    trainCr = CorpusReader(filepath1)
    testCr = CorpusReader(filepath2)

    trainX = []
    trainY = []
    testX = []
    testY = []

    for inp in trainCr.getInputArray():
        # print(inp.sentence1)
        # print(inp.sentence2)

        trainX.append(Input.similarityMatrix(inp.sentence1, inp.sentence2))
        # print(inp.id,trainX[-1].shape)
        trainY.append(inp.score)

    for inp in testCr.getInputArray():
        testX.append(Input.similarityMatrix(inp.sentence1, inp.sentence2))
        testY.append(inp.score)

    # print(len(trainX))
    # print(len(trainY))

    model = svm.SVC()
    model.fit(trainX, trainY)
    predictedScores = model.predict(testX)
    print(accuracy_score(testY, predictedScores))



