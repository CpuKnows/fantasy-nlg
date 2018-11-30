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

tagIdDict = {}
inverseTagIdDict = {}  # Used for lookup for resolving later
bucket = []
rgx = re.compile('\${.*?}')
tagId = 0
for i in range(len(df)):
    currentBucket = []
    for col in my_cols:
        chunk = df[col][i]
        if pd.isnull(chunk): continue
        tags = rgx.findall(chunk)
        if len(tags) > 0:
            tag = ''.join(tags)
            if tag in tagIdDict:
                tagId = tagIdDict[tag]
            else:
                tagId += 1
                tagIdDict[tag] = tagId
                inverseTagIdDict[tagId] = tag
            currentBucket.append(tagId)
    bucket.append(currentBucket)

def openResultFile():
    result_file = "result.txt";

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
    for n, Fk in enumerate(Fks):
        if len(Fk) == 0: continue

        k = len(Fk[0])
        file.write("\nFrequent %d-tagsets\n" % k)
        counter = 0
        for i, itemset in enumerate(Fk):
            file.write("%d : %s\n" % (item_counts[n][i], itemset))
            counter += 1

    file.write("\nTotal no. of frequent-%d tagsets = %d\n" % (k, counter))

minsup = 0.4
Fks, item_counts = MSapriori(bucket, minsup)

file = openResultFile()
file.write("Generating frequent tagsets for minsup: % 0.2f" % minsup)
printFrequentkItemsets(Fks, item_counts, file)
resolveFks(Fks)
printFrequentkItemsets(Fks, item_counts, file)
file.close()



