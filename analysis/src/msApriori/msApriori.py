''' 
Author: vsingh38@uic.edu
Calculates frequenst item sets of Transactions bucket using MsApriori algorithm.
Multiple min-sup has been skipped for this implementation. 
Pass global min-sup to MsApriori() as second argument.
Tail counts are calculated but not used as of now. Can be used if needed. See MsApriori().
Sorting of items in bucket also ignored to maintain original order.
'''
def supportCount(itemset, T):    # returns count of itemset in T
    count = 0

    # Increment count if itemset in t
    for t in T:
        is_in_t = len(itemset) > 0
        for i in itemset:
            is_in_t = is_in_t and i in t
        if is_in_t:
            count += 1
    
    return count

support = lambda count, T: float(count) / len(T)

def initPass(M, T, MS):
    l = []
    l_count = []    # to memoize the support count of l

    for i,m in enumerate(M):
        sup_count = supportCount([m], T)

        if support(sup_count, T) >= MS:
            # Add m to l, and all subsequent j"s if sup(j) >= MS
            l.append(m)
            l_count.append(sup_count)

            for j in M[i + 1:]:
                sup_count = supportCount([j], T)

                if support(sup_count, T) >= MS:
                    l.append(j)
                    l_count.append(sup_count)
            break
    return l, l_count

def kLess1Subsets(c):
    subsets = []
    for i in range(len(c)):
        subsets.append([x for i_,x in enumerate(c) if i_ != i])
    return subsets

def MScandidate_gen(Fk_less1):
    C = []
    n = len(Fk_less1)
    nk = len(Fk_less1[0])    # Get size of k-1

    for i in range(n):
        for j in range(n):
            if i == j: continue

            f1 = Fk_less1[i]    # TODO: verify if the swap needs to be considered as well.
            f2 = Fk_less1[j]

            if f1[0:nk - 1] == f2[0:nk - 1] and f1[-1] < f2[-1]:
                c = [x for x in f1]     # Deep copy
                c.append(f2[-1])        # f1 U f2

                # Check pruning condition before appending
                should_append = True
                for s in kLess1Subsets(c):
                    if s not in Fk_less1:
                        should_append = False

                # Check constraints
                if should_append:
                    C.append(c)
    return C

def printFrequentkItemsets(Fks, item_counts):
    for n, Fk in enumerate(Fks):
        if len(Fk) == 0: continue

        k = len(Fk[0])
        print("\nFrequent %d-itemsets\n" % k)
        counter = 0
        for i, itemset in enumerate(Fk):
            print("%d : %s\n" % (item_counts[n][i], itemset))
            counter += 1

    print("\nTotal no. of frequent-%d itemsets = %d\n" % (k, counter))
 
def MSapriori(T, MS):
    frequentSets = []
    itemCounts = []
    temp_s = set()
    for t in T: 
        for x in t: temp_s.add(x)
    
    M = list(temp_s)     # Uses Python < 3.x
    L, L_counts = initPass(M, T, MS)
    Fk = []   # F_k-1
    Fk1_counts = []
    F = []      # For union over all Fk

    # Populate F1
    for i,x in enumerate(L):
        if support(L_counts[i], T) >= MS:
            Fk.append([x])
            Fk1_counts.append(L_counts[i])

    frequentSets.append(Fk)
    itemCounts.append(Fk1_counts)

    k = 2
    while len(Fk) > 0:
        C = MScandidate_gen(Fk)
        c_counts = []
        tail_counts = []

        for c in C:
            c_counts.append(supportCount(c, T))
            tail_counts.append(supportCount(c[1:], T))  # Persist this in global array if rule gen reqd.

        Fk = []
        c_counts_temp = []
        t_counts_temp = []

        for i,c in enumerate(C):
            if support(c_counts[i], T) >= MS:
                Fk.append(c)
                c_counts_temp.append(c_counts[i])
                t_counts_temp.append(tail_counts[i])
        frequentSets.append(Fk)
        itemCounts.append(c_counts_temp)
        [F.append(x) for x in Fk if x not in F]
        k += 1
    return frequentSets, itemCounts


# Uncomment for testing
#T = [[20, 30, 80, 70, 50, 90],
#    [20, 10, 80, 70],
#    [10, 20, 80],
#    [20, 30, 80],
#    [20, 80],
#    [20, 30, 80, 70, 50, 90, 100, 120, 140]]

#Fks, counts = MSapriori(T, 0.5)
#printFrequentkItemsets(Fks, counts)
