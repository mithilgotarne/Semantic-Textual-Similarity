import Input

class CorpusReader:

    def __init__(self, filepath):

        self.inputObjects = {}

        with open(filepath) as fp:
            line =  fp.readline()

            while line:
                line = fp.readline()
                id, s1, s2, score = line.split('\t')
                inputObject = Input(id, s1, s2, score)
                self.inputObjects[id] = inputObject

    def getInputArray(self):
        return self.inputObjects.values()

    def getInput(self, id):
        return self.inputObjects[id]


