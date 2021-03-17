"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Implementation of One-bit Purifying Search (OPS)
"""

from evaluation import evaluation, features
from bas import Ant
import random
import numpy as np
from operator import attrgetter



def OPS(population, x, y):
    """
        Function name : OPS
        Arguments : 
            -- population : Binary ant system population
            -- x : input features
            -- y : input types/classes
    """

    # find best solution (max fitness value)
    best = Ant(len(population[0].solution))
    for i in range(len(population)):
        if best.fitness <= population[i].fitness:
            best = population[i]
    
    # for finding u1
    temp_u1 = np.where(np.array(best.solution)==1)[0]
    u1 = temp_u1[random.randint(0,len(temp_u1)-1)]

    # for finding u2
    temp_u2 = np.where(np.array(best.solution)==0)[0]
    u2 = temp_u2[random.randint(0,len(temp_u2)-1)]

    # solutions required to compare
    X1 = best.solution
    
    X2 = X1
    X2[u1] = 0
    X2[u2] = 1

    X3 = X1
    X3[u1] = 0

    # calculate fitness values
    f1 = evaluation(features(x,X1),y)
    f2 = evaluation(features(x,X2),y)
    f3 = evaluation(features(x,X3),y)

    # find which of u1, u2 is more important
    if abs(f1 - f3) >= abs(f2 - f3):
        u_imp = u1
        u_notimp = u2
    else:
        u_imp = u2
        u_notimp = u1

    # do for all solutions in the population
    newP = list()
    for i in range(len(population)):
        newS = population[i]
        newSol = newS.solution
        # check all four cases
        if newSol[u_imp] == 0 and newSol[u_notimp] == 0:
            newSol[u_imp] = 1
            newS.numFeaturesSelected += 1
        elif newSol[u_imp] == 0 and newSol[u_notimp] == 1:
            newSol[u_imp] = 1
            newSol[u_notimp] = 0
        elif newSol[u_imp] == 1 and newSol[u_notimp] == 0:
            newSol[u_imp] = 0
            newS.numFeaturesSelected -= 1
        elif newSol[u_imp] == 1 and newSol[u_notimp] == 1:
            newSol[u_notimp] = 0
            newS.numFeaturesSelected -= 1
        
        # evaluate the solutions
        population[i].fitness = evaluation(features(x,population[i].solution),y)
        newS.fitness = evaluation(features(x,newSol),y)

        # add both to new population
        newP.append(population[i])
        newP.append(newS)

    # sort and remove extra members of the population
    for i in range(len(newP)):
        for j in range(i+1,len(newP)):
            if newP[i].fitness < newP[j].fitness:
                newP[i], newP[j] = newP[j], newP[i]
    newP = newP[:len(population)]

    return newP