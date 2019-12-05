import nltk
from collections import OrderedDict
from nltk.corpus import wordnet as wordnet
import spacy

lemmatizer = nltk.stem.WordNetLemmatizer()
model = spacy.load("en_core_web_sm")


class Sentence:
    def __init__(self, string):
        self.string = string
        self.tokens = self.tokenize()
        self.lemmatizedTokens = self.lemmatize()
        self.pos_tags = self.POSTags()
        self.wordToHyponyms, self.wordToHypernyms, self.wordToPartsMeronyms, self.wordToSubstanceMeronyms, self.wordToPartsHolonyms, self.wordToSubstanceHolonyms = (
            self.wordnetComponents()
        )
        self.dependencyTriples = self.generateDependencyParseTree(self.string)

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
    def generateDependencyParseTree(self, string=None):
        dependencyTriples = []
        if not string:
            string = self.string
        tree = model(string)
        for token in tree:
            dependencyTriples.append((token.text, token.pos_, token.dep_))
        return dependencyTriples

    # Synonymns and antonyms/
    # WUP similarity


class Input:
    def __init__(self, id, s1, s2, score):
        self.id = id
        self.sentence1 = Sentence(s1)
        self.sentence2 = Sentence(s2)
        self.score = score


if __name__ == "__main__":

    import sys

    s = Sentence(sys.argv[1])
    print(s)
