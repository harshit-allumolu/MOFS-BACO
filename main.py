"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Implementation of MOFS-BACO (Binary Ant Colony Optimization 
    with Self-learning for Multi Objective Feature Selection)
"""

from bas import Ant, BinaryAntSystem
from evaluation import evaluation
import pandas as pd
import numpy as np


def MOFS_BACO(numFeatures, x, y):
    """
        Function name : MOFS-BACO
        Arguments : 
            -- numFeatures : The total number of features in the dataset
            -- x : input features
            -- y : input types/classes
    """
    
    # intialization
    graph = BinaryAntSystem(numFeatures)
    P = 10              # OPS interval
    iterations = 10   # tuning required
    lambda_ = 0.01      # lambda value
    Supd = [
        Ant(numFeatures),
        Ant(numFeatures),
        Ant(numFeatures)
    ]   # ib, rb and gb respectively

    # loop start
    for i in range(iterations):
        # generate ants (new generation)
        graph.generateNewAnts()
        
        # construct solutions
        graph.traverse()

        # OPS periodic condition
        if (i+1) % P == 0:
            pass        # add ops here
        
        # solution evaluation
        for i in range(graph.m):
            x_temp = features(x,graph.population[i].solution)
            # fitness function is f = accuracy / (1 + lambda * #features)
            graph.population[i].fitness = evaluation(x_temp,y) / (1 + lambda_ * graph.population[i].numFeaturesSelected)

        best = Ant(numFeatures)
        for i in range(graph.m):
            # error less than best => make it best
            if graph.population[i].fitness > best.fitness:
                best = graph.population[i]         
            # error == best => make it best if less number of features are selected
            elif graph.population[i].fitness == best.fitness:
                if graph.population[i].numFeaturesSelected < best.numFeaturesSelected:
                    best = graph.population[i]
        
        Supd[0] = best
        if best.fitness > Supd[1].fitness:
            Supd[1] = best
        if best.fitness > Supd[2].fitness:
            Supd[2] = best
        
        # calculate convergence factor
        graph.convergenceFactor()   # graph.cf

        # update pheromone values
        graph.updatePheromone(Supd)

    return Supd[2].solution  # global best solution



def features(x, solution):
    """
        Function name : features
        Arguments : 
            -- x : Input features
            -- solution : A binary array to select features
        Returns : 
            -- x_temp : The selected input features
    """

    temp = []
    for i in range(len(solution)):
        if solution[i] == 1:
            temp.append(x[:,i:i+1])
    x_temp = np.concatenate(temp,axis=1)
    return x_temp



if __name__ == "__main__":
    # read dataset
    filename = "datasets/" + input("Enter the dataset name : ")
    dataset = pd.read_csv(filename)
    y = dataset.iloc[:,0].to_numpy()
    x = dataset.iloc[:,1:].to_numpy()

    # run the algorithm
    best = MOFS_BACO(len(x[0]),x,y)
    f = features(x,best)
    acc = evaluation(f,y)
    print("\n\nNumber of features selected = {}".format(best.count(1)))
    print("Accuracy = {}%\n\n".format(round(acc*100,2)))