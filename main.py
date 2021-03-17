"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Implementation of MOFS-BACO (Binary Ant Colony Optimization 
    with Self-learning for Multi Objective Feature Selection)
"""

from bas import Ant, BinaryAntSystem
from evaluation import evaluation
from evaluation import features
from ops import OPS
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
    P = 5              # OPS interval
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
            OPS(graph.population, x, y)        # add ops here
        
        else:
            # solution evaluation
            for i in range(graph.m):
                x_temp = features(x,graph.population[i].solution)
                # fitness function is f = accuracy / (1 + lambda * #features)
                graph.population[i].fitness = evaluation(x_temp,y)

        best = Ant(numFeatures)
        for i in range(graph.m):
            # fitness >= best
            if graph.population[i].fitness >= best.fitness:
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




if __name__ == "__main__":
    # read dataset
    filename = "datasets/" + input("Enter the dataset name : ")
    dataset = pd.read_csv(filename)
    y = dataset.iloc[:,0].to_numpy()
    x = dataset.iloc[:,1:].to_numpy()

    lambda_ = 0.01      # tuning required

    # run the algorithm
    best = MOFS_BACO(len(x[0]),x,y)
    f = features(x,best)
    acc = evaluation(f,y) * (1 + lambda_ * best.count(1))
    print("\n\nNumber of features selected = {}".format(best.count(1)))
    print("Accuracy = {}%\n\n".format(round(acc*100,2)))