import random
import numpy as np
from matplotlib import pyplot as plt 

global backmoves 
backmoves = 0
global localmoves
localmoves = 0

def backtrack(a, b):
    global backmoves
    if(isConsistent(a,b)):
        if(len(a) < len(b[0])):
            for i in range(len(a),len(b[0])):
                a.append(-1)
        return a
    elif(len(a) == len(b[0])): #Length is equal and not consistent
        return []
    else:
        vect = a.copy()
        #print(len(vect))
        v = [-1,1] #We don't assign 0. 
        #empt = a
        for x in v:
            vect.append(x)
            #print(vect)
            #if isConsistent(vect,b):
            backmoves += 1
            result = backtrack(vect,b)
                #print(vect)
            if len(result) > 0:
                return result
            vect = vect[:-1]
    return []

def local_search(a, b):
    #Can instead see if numConflicts == 0
    global localmoves
    if(localmoves == 0):
        a = generate_random_vector(len(b[0]))
    #backtracking at most requires (2^n+1)-2 moves to determine no solution. The division is to keep running time similar 
    #between the two algorithms
    while(localmoves <= (2**(len(b[0])+1))/len(b[0])):
        if(isConsistent(a,b)):
            return (a)
        a = pick_best_successor(a,b) #if no better successor, pick best successor returns a new
        localmoves += 1
    if(localmoves > (2**(len(b[0])+1))/len(b[0])):
        return ([])
    #random vector that local search then tries


def pick_best_successor(a, b):
    #Successor is not really required - code can easily be refactored to remove it, but this is left for later,
    #for want of time (because bugs may pop up when substantial modifications are made to the code)
    successor = [[0 for j in range(2)] for i in range(len(a)+1)]
    successor[0][0] = numconflicts(a,b)
    successor[0][1] = a.copy()
    k = 0
    low = numconflicts(a,b)
    for i in range(len(a)):
        vect = a.copy()
        vect[i] = -a[i]
        #print(vect)
        successor[i+1][0] = numconflicts(vect,b)
        successor[i+1][1] = vect.copy()
        if(successor[i+1][0] < low):
            low = successor[i+1][0]
            k = i+1
    if(k != 0):
        return successor[k][1]
    else:
        here = generate_random_vector(len(a))
        #print(here)
        return here

def numconflicts(vect, b):
    conflicts = len(b)
    for j in range (0,len(b)):
        for i in range(0,len(vect)):
            if(vect[i] == b[j][i]): # or b[j][i] == 0
                conflicts -= 1
                break
    return conflicts


def generate_random_vector(n):
    l = [-1,1]
    vect = []
    for i in range(0,n):
        vect.append(random.choice(l))
    return vect

def generate_random_maxtrix(m,n,p):
    l = [-1,0,1]
    ar = np.zeros((m,n))
    #ar = np.random.choice([-1,0,1],size=(m,n), p=[p[0],p[1],p[2]])
    # Just generate the rows, make sure they are not 0 rows, and put them into matrix
    count = 0
    iter = 0
    for i in range(m):
        ar[i] = np.random.choice([-1,0,1],size=(n), p=[p[0],p[1],p[2]])
        while(np.count_nonzero(ar[i]) == 0):
            ar[i] = np.random.choice([-1,0,1],size=(n), p=[p[0],p[1],p[2]])
    return ar


def isConsistent(a, b):
    counter = 0
    for j in range (0,len(b)):
        for i in range(0,len(a)):
            if(a[i] == b[j][i]): # or b[j][i] == 0
                counter += 1
                break
                #Return a vector consisting of booleans or something to signify which ones worked
        #print(x)
    #print(counter,a)
    if counter == len(b):
        return True
    return False

def main():
    a = []
    bmoves = []
    lmoves = []
    global backmoves
    global localmoves
    for i in range(5,16):
        b = generate_random_maxtrix(200,i,[0.4,0.2,0.4])#(1.0-i/10.0)/2
        backtrack(a,b)
        local_search(a,b)
        bmoves.append(backmoves)
        lmoves.append(localmoves)
        backmoves = 0
        localmoves = 0
        #print(backtrack(a,b),backmoves)
        #print(local_search(a,b),localmoves)
    print(bmoves)
    print(lmoves)
    #Histograms didn't represent the findings so well, and have been left for investigation in future assignments. 
    #plt.hist(bmoves, bins = [0,5,10,15,20,40]) 
    #plt.show()
    #lhist = np.histogram(lmoves)
    #print(lhist)
    #print(b)
    #vect = backtrack(a,b)
    #print(backtrack(a,b),backmoves)
    #print(local_search(a,b),localmoves)

if __name__ == "__main__":
    main()