from msApriori import MSapriori
import numpy as np
import pandas as pd
import os
import re

filepath = "../../data/output_chunked.csv"

# Workaround to load variable length rows using pandas (missing values will be
# NaN)
my_cols = ["A", "B", "C", "D", "E", "F", "G"]
df = pd.read_csv(filepath, encoding = "ISO-8859-1", names=my_cols)

tagChunksDict = {}  # Stores chunks that the tags belong
def addTagToChunkDict(tag, chunk):
    if tag in tagChunksDict:
        # Check if this chunk already added
        if chunk not in tagChunksDict[tag]:
            tagChunksDict[tag].append(chunk)
    else:
        tagChunksDict[tag] = [chunk]

tagIdDict = {}
inverseTagIdDict = {}  # Used for lookup for resolving later
bucket = []
rgx = re.compile('\${.*?}')
tagId = 0
lastTagId = 0
for i in range(len(df)):
    currentBucket = []
    for col in my_cols:
        chunk = df[col][i]
        if pd.isnull(chunk): continue
        tags = rgx.findall(chunk)
        if len(tags) > 0:
            tag = ''.join(tags)
            addTagToChunkDict(tag, chunk)
            if tag in tagIdDict:
                tagId = tagIdDict[tag]
            else:
                lastTagId += 1
                tagId = lastTagId
                tagIdDict[tag] = tagId
                inverseTagIdDict[tagId] = tag
            currentBucket.append(tagId)
    bucket.append(currentBucket)

def openResultFile():
    result_file = "result.txt"

    try:
        os.remove(result_file)
    except OSError:
        pass

    return open(result_file, "w") 

# Resolves tag ids to tags
def resolveFks(Fks):
    # Brute-force the 3D array for now
    for i in range(len(Fks)):
        for j in range(len(Fks[i])):
            for k in range(len(Fks[i][j])):
                Fks[i][j][k] = inverseTagIdDict[Fks[i][j][k]]

def printFrequentkItemsets(Fks, item_counts, file):
    for j, Fk in enumerate(Fks):
        if len(Fk) == 0: continue

        k = len(Fk[0])
        file.write("\nFrequent %d-tagsets\n" % k)
        counter = 0
        for i, itemset in enumerate(Fk):
            file.write("%d : %s\n" % (item_counts[j][i], itemset))
            counter += 1

    file.write("\nTotal no. of frequent-%d tagsets = %d\n" % (k, counter))

def printTagChunks(tagChunksDict, file):
    file.write("\nPrinting tag-chunks dictionary..\n")
    for key in tagChunksDict.keys():
        file.write("\n" + key + ": \n")
        file.write("----------------\n")
        for val in tagChunksDict[key]:
            file.write(val + "\n")

minsup = 0.3
Fks, item_counts = MSapriori(bucket, minsup)

file = openResultFile()
file.write("Generating frequent tagsets for minsup: % 0.2f" % minsup)
printFrequentkItemsets(Fks, item_counts, file)
resolveFks(Fks)
printFrequentkItemsets(Fks, item_counts, file)
printTagChunks(tagChunksDict, file)
file.close()

#---------------------------
def generateChunkCombinations(Fks, k, tagChunksDict):
    # Verify if fks have k-frequent sets
    k_fk = None
    
    for j, Fk in enumerate(Fks):
        if len(Fk) == 0: continue
        if k == len(Fk[0]):
            k_fk = Fk
            break

    if k_fk == None:
        raise("k-frequent set not found in Fks.")

    sentences = []
    for fk in k_fk:
        currentSentences = []
        combineChunks(fk, tagChunksDict, 0, '', currentSentences)
        sentences.append(currentSentences)
    return sentences

def combineChunks(fk, chunksDict, chunkIndex, predecessor, sentences):
    if chunkIndex == len(fk):
        sentences.append(predecessor)
        return

    for chunk in chunksDict[fk[chunkIndex]][:5]:
        currentSentence = predecessor + " " + chunk
        combineChunks(fk, chunksDict, chunkIndex + 1, currentSentence, sentences)

from pprint import pprint
#pprint(generateChunkCombinations(Fks, 4, tagChunksDict))