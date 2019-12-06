import nltk
from collections import OrderedDict
from nltk.corpus import wordnet as wordnet
import spacy
from nltk.wsd import lesk
from nltk.parse.corenlp import CoreNLPDependencyParser
import numpy as np
from nltk.corpus import stopwords
from nltk import Tree


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
        self.generateDependencyParseTreeSpacy(self.string)
        self.tokenToMostProbableSynset = self.mostProbableSynset(
            self.string, self.tokens
        )
        self.tree = self.generateDependencyParseTreeSpacy(self.string)

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
        if not string:
            string = self.string
        tree = spacyModel(string)
        # for token in tree:
        #     dependencyTriples.append((token.text, token.pos_, token.dep_))
        # print("Root is ", tree.head.text, " And dependency is  ", tree.dep)
        importantContext = set()
        for eachSubjectObject in tree:
            if(eachSubjectObject.pos == "NOUN" or eachSubjectObject.pos == "VERB"):
                importantContext.add(eachSubjectObject.text)
        
        # for each in tree:
        #     # print("\n\nRoot is ", each.text, " And dependency is  ", each.dep_)
        #     for child in each.children:
        #         print("The child is ", child.text, child.pos_, child.dep_)
        # [to_nltk_tree(each.root).pretty_print() for each in tree.sents]
        return tree

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
        # if token not in stopWords:
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

#Remove this.
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [(node.pos,to_nltk_tree(child)) for child in node.children])
    else:
        return node.orth_

def getRoot(tree):
    print(dir(tree))
    for eachEntry in tree:
        #find the root and stop
        if(eachEntry.dep_ == "ROOT"):
            dir(eachEntry)
            return eachEntry
    return None
    

def createDepParserFeatures(sentence1, sentence2):
    treeObject1, treeObject2 = sentence1.tree, sentence2.tree
    root1 = getRoot(treeObject1)
    root2 = getRoot(treeObject2)

    featureMatrix = []
    traverseTree(root1, root2, sentence1, sentence2, 0, featureMatrix)
    # featureMatrix2 = []
    # traverseTree(root1, root2, sentence2, sentence1, 0, featureMatrix2)
    print(featureMatrix)




def traverseTree(root1, root2, sentence1, sentence2, level, featureMatrix):
    print(root1.text)
    if root1:
        answer = [float("-inf"), float("-inf")]
        searchInOtherTree(root1.text, root2, level, sentence1, sentence2, answer)
        if answer is not [float("-inf"), float("-inf")]:
            featureMatrix.append(answer)
        for eachChild in root1:
            traverseTree(eachChild.text, root2, sentence1, sentence2, level + 1, featureMatrix)


        
def searchInOtherTree(searchString, treeObject, level, sentence1, sentence2, resultList):

    string1 = treeObject.text
    synset1 = synset2 = None
    if searchString in sentence1.tokenToMostProbableSynset:
        synset1 = sentence1.tokenToMostProbableSynset[searchString]
    if string1 in sentence2.tokenToMostProbableSynset:
        synset2 = sentence2.tokenToMostProbableSynset[string1]
    if synset1 and synset2:
        similarity = synset1.wup_similarity(synset2)
        if similarity > resultList[0]:
            resultList[0] = similarity
            resultList[1] = level
    for eachChild in treeObject.children:
        searchInOtherTree(searchString, eachChild, level+1, sentence1, sentence2, resultList)


if __name__ == "__main__":
    import sys

    # s = Sentence(sys.argv[1])
    string1 = Sentence("We often pontificate here about being the representatives of the citizens of Europe.")
    string2 = Sentence("We are proud often here to represent the citizens of Europe.")
    # print(string1.tokenToMostProbableSynset)
    # print(string2.tokenToMostProbableSynset)
    createDepParserFeatures(string1, string2)
    # print(s)

