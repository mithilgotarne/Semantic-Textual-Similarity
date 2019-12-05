import nltk
from collections import OrderedDict
from nltk.corpus import wordnet as wordnet
import spacy
from nltk.wsd import lesk
from nltk.parse.corenlp import CoreNLPDependencyParser
import numpy as np
from nltk.corpus import stopwords


stopWords = set(stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
spacyModel = spacy.load("en_core_web_lg")
# stanfordParser = r'C:\Users\chitt\Desktop\Fall2019_Semester\Natural Language Processing\stanford-parser-full-2018-10-17\stanford-parser-full-2018-10-17\stanford-parser.jar'
# modelJar = r'C:\Users\chitt\Desktop\Fall2019_Semester\Natural Language Processing\stanford-english-corenlp-2018-10-05-models.jar'
# dependency_parser = CoreNLPDependencyParser(url='http://localhost:9000')


class Sentence:
    def __init__(self, string):
        self.string = string
        self.tokens = self.tokenize()
        # self.lemmatizedTokens = self.lemmatize()
        # self.pos_tags = self.POSTags()
        # self.wordToHyponyms, self.wordToHypernyms, self.wordToPartsMeronyms, self.wordToSubstanceMeronyms, self.wordToPartsHolonyms, self.wordToSubstanceHolonyms = (
        #     self.wordnetComponents()
        # )
        # self.dependencyTriples = self.generateDependencyParseTree(self.string)
        self.tokenToMostProbableSynset = self.mostProbableSynset(
            self.string, self.tokens
        )

    def __str__(self):
        return (
            "string: "
            + str(self.string)
            + "\n\ntokens: "
            + str(self.tokens)
            + "\n\nlemmatizedTokens: "
            + str(self.lemmatizedTokens)
            + "\n\npos_tags"
            + str(self.pos_tags)
            + "\n\nHyponymns"
            + str(self.wordToHyponyms)
            + "\n\nHypernyms: "
            + str(self.wordToHypernyms)
            + "\n\nParts Meronyms: "
            + str(self.wordToPartsMeronyms)
            + "\n\nSubstance Meronyms: "
            + str(self.wordToSubstanceMeronyms)
            + "\n\nParts Holonyms: "
            + str(self.wordToPartsHolonyms)
            + "\n\nSubstance Holonyms: "
            + str(self.wordToSubstanceHolonyms)
            + "\n\nDependecy Triples: "
            + str(self.dependencyTriples)
            + "\n\nMost Probable Synset: "
            + str(self.tokenToMostProbableSynset)
            + "\n\n"
        )

    def tokenize(self, string=None):
        if not string:
            string = self.string
        return nltk.word_tokenize(string)

    def POSTags(self, tokens=None):
        if not tokens:
            tokens = self.tokens
        return nltk.pos_tag(tokens)

    def lemmatize(self, tokens=None):
        lemmatizedTokens = OrderedDict()
        if not tokens:
            tokens = self.tokens

        for token in tokens:
            lemmatizedTokens[token] = lemmatizer.lemmatize(token)

        return lemmatizedTokens

    def wordnetComponents(self, tokens=None):
        if not tokens:
            tokens = self.tokens
        wordToHyponyms = OrderedDict()
        wordToHypernyms = OrderedDict()
        wordToPartsMeronyms = OrderedDict()
        wordToSubstanceMeronyms = OrderedDict()
        wordToPartsHolonyms = OrderedDict()
        wordToSubstanceHolonyms = OrderedDict()
        for eachToken in tokens:
            synSets = wordnet.synsets(eachToken)
            for eachSynSet in synSets:
                eachInterpretation = wordnet.synset(eachSynSet.name())
                self.populateWordnetComponents(
                    eachInterpretation.hyponyms(), eachToken, wordToHyponyms
                )
                self.populateWordnetComponents(
                    eachInterpretation.hypernyms(), eachToken, wordToHypernyms
                )
                self.populateWordnetComponents(
                    eachInterpretation.part_meronyms(), eachToken, wordToPartsMeronyms
                )
                self.populateWordnetComponents(
                    eachInterpretation.substance_meronyms(),
                    eachToken,
                    wordToSubstanceMeronyms,
                )
                self.populateWordnetComponents(
                    eachInterpretation.part_holonyms(), eachToken, wordToPartsHolonyms
                )
                self.populateWordnetComponents(
                    eachInterpretation.substance_holonyms(),
                    eachToken,
                    wordToSubstanceHolonyms,
                )
        return (
            wordToHyponyms,
            wordToHypernyms,
            wordToPartsMeronyms,
            wordToSubstanceMeronyms,
            wordToPartsHolonyms,
            wordToSubstanceHolonyms,
        )

    def populateWordnetComponents(
        self, wordnetComponentData, eachInterpretation, dictionaryToPopulate
    ):
        for eachItem in wordnetComponentData:
            for lemma in eachItem.lemmas():
                if eachInterpretation in dictionaryToPopulate:
                    dictionaryToPopulate[eachInterpretation].append(lemma.name())
                else:
                    dictionaryToPopulate[eachInterpretation] = [lemma.name()]

    # Dependency Parse tree using Spacy
    def generateDependencyParseTreeSpacy(self, string=None):
        dependencyTriples = []
        if not string:
            string = self.string
        tree = model(string)
        for token in tree:
            dependencyTriples.append((token.text, token.pos_, token.dep_))
        return dependencyTriples

    def generateDependencyParseTree(self, string=None):
        if not string:
            string = self.string

        parse, = dependency_parser.raw_parse(string)
        return list(parse.triples())
        # print(parse)
        # return list(parse.triples())

    # Most probable synset
    def mostProbableSynset(self, string, tokens):
        tokenToMostProbableSynset = OrderedDict()
        for token in tokens:
            if token not in stopWords:
                mostProbable = lesk(string, token)
                if mostProbable:
                    tokenToMostProbableSynset[token] = mostProbable
        return tokenToMostProbableSynset

    # Synonymns and antonyms/
    # WUP similarity


class Input:
    def __init__(self, id, s1, s2, score):
        self.id = id
        self.sentence1 = Sentence(s1)
        self.sentence2 = Sentence(s2)
        self.score = int(score)


def similarityMatrix(sentence1, sentence2):
    dim = max(
        len(sentence1.tokenToMostProbableSynset),
        len(sentence2.tokenToMostProbableSynset),
    )

    # matrix = np.zeros(shape=(3, len(sentence1.tokenToMostProbableSynset), len(sentence2.tokenToMostProbableSynset)))

    matrix = np.zeros(shape=(3, dim, dim))

    # matrix = np.zeros(shape=(3, 100, 100))

    for i, synset1 in enumerate(sentence1.tokenToMostProbableSynset.values()):
        for j, synset2 in enumerate(sentence2.tokenToMostProbableSynset.values()):
            pathSimilarity = synset1.path_similarity(synset2)

            lchSimilarity = None
            if synset1.pos() == synset2.pos():
                lchSimilarity = synset1.lch_similarity(synset2)

            wupSimilarity = synset1.wup_similarity(synset2)
            if pathSimilarity:
                matrix[0][i][j] = pathSimilarity
            if lchSimilarity:
                matrix[1][i][j] = lchSimilarity
            if wupSimilarity:
                matrix[2][i][j] = wupSimilarity

    doc1 = spacyModel(sentence1.string)
    doc2 = spacyModel(sentence2.string)

    return (
        np.linalg.det(matrix[0]),
        np.linalg.det(matrix[1]),
        np.linalg.det(matrix[2]),
        doc1.similarity(doc2),
    )


if __name__ == "__main__":
    import sys

    s = Sentence(sys.argv[1])
    print(s)
