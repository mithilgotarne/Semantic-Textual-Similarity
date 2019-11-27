import sys, nltk, CorpusReader

def tokenize(s):
    return nltk.word_tokenize(s)

if __name__ == "__main__":
    filepath = sys.argv[0]
    cr = CorpusReader(filepath)

    for inputObject in cr.getInputArray():
        words = tokenize(inputObject.sentence1)
        lemmas = lemmatize(words)

