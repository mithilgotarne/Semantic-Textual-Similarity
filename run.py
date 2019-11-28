import sys, nltk
from CorpusReader import CorpusReader


if __name__ == "__main__":
    filepath = sys.argv[1]
    cr = CorpusReader(filepath)

    for inp in cr.getInputArray():
        print(inp.sentence1)
        print(inp.sentence2)







