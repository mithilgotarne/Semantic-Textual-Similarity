import nltk
from collections import OrderedDict

lemmatizer = nltk.stem.WordNetLemmatizer()

class Sentence:
    def __init__(self, string):
        self.string = string
        self.tokens = self.tokenize()
        self.lemmatizedTokens = self.lemmatize()
        self.pos_tags = self.POSTags()

    def __str__(self):
        return "string: " + str(self.string) + "\ntokens: " + str(self.tokens) + "\nlemmatizedTokens: " + str(self.lemmatizedTokens) + "\npos_tags" + str(self.pos_tags)

    def tokenize(self, string=None):
        if not string:
            string = self.string
        return nltk.word_tokenize(string)

    def POSTags(self, tokens=None):
        if not tokens:
            string = self.string
        return nltk.pos_tag(string)

    def lemmatize(self, tokens=None):
        lemmatizedTokens = OrderedDict()
        if not tokens:
            tokens = self.tokens

        for token in tokens:
            lemmatizedTokens[token] = lemmatizer.lemmatize(token)

        return lemmatizedTokens


class Input:
    def __init__(self, id, s1, s2, score):
        self.id = id
        self.sentence1 = Sentence(s1)
        self.sentence2 = Sentence(s2)
        self.score = score

