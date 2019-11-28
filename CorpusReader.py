import csv
from Input import Input


class CorpusReader:
    def __init__(self, filepath):

        self.inputObjects = {}

        with open(filepath) as fp:

            reader = csv.reader(fp, delimiter="\t")
            next(reader)

            for row in reader:
                id, s1, s2, score = row
                inputObject = Input(id, s1, s2, score)
                self.inputObjects[id] = inputObject

    def getInputArray(self):
        return self.inputObjects.values()

    def getInput(self, id):
        return self.inputObjects[id]

