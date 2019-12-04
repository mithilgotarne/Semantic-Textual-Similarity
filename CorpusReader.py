import csv
from Input import Input
class CorpusReader:
    def __init__(self, filepath):

        self.inputObjects = {}

        with open(filepath, encoding="utf8") as fp:

            next(fp)

            for line in fp:
                line = line.strip()
                id, s1, s2, score = line.split('\t')
                inputObject = Input(id, s1, s2, score)
                self.inputObjects[id] = inputObject
        print(self.inputObjects["s_1"])
    def getInputArray(self):
        return self.inputObjects.values()

    def getInput(self, id):
        return self.inputObjects[id]

