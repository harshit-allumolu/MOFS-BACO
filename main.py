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


def MOFS_BACO(numFeatures, x, y, iterations=200, P=5, lambda_=0.01, m=20, ro=0.02,k=1):
    """
        Function name : MOFS-BACO
        Arguments : 
            -- numFeatures : The total number of features in the dataset
            -- x : input features
            -- y : input types/classes
            -- iterations : Number of iterations (default = 200)
            -- P : time period for ops (default = 5)
            -- lambda_ : A variable to control the effect of #features in fitness function
            -- m : Number of ants
            -- ro : evaporation factor
            -- k : k in knn
    """
    
    # intialization
    graph = BinaryAntSystem(numFeatures,m=m,ro=ro)
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
            OPS(graph.population, x, y, lambda_,k)        # add ops here
        
        else:
            # solution evaluation
            for i in range(graph.m):
                x_temp = features(x,graph.population[i].solution)
                # fitness function is f = accuracy / (1 + lambda * #features)
                graph.population[i].fitness = evaluation(x_temp,y,lambda_,k)

        best = Ant(numFeatures)
        for i in range(graph.m):
            # fitness >= best
            if graph.population[i].fitness >= best.fitness:
                best = graph.population[i]         
        
        Supd[0] = best #Iteration best
        if best.fitness > Supd[1].fitness and graph.re_init == 0: #If re-initialization not happened then compare ib with rb.
            Supd[1] = best
        elif graph.re_init == 1: #If re-initialization has happened then rb = ib and set re_init flag to zero again.
            Supd[1] = best
            graph.re_init = 0
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

    # all the variable which require tuning
    n_iterations = 30   # number of iterations
    P = 5               # time period for ops
    lambda_ = 0.01      # tuning required
    m = 20              # number of ants
    ro = 0.02           # evaporation factor
    k = 1               # k in knn

    # run the algorithm
    best = MOFS_BACO(len(x[0]),x,y,iterations=n_iterations,P=P,lambda_=lambda_,m=m,ro=ro,k=k)
    f = features(x,best)
    acc = evaluation(f,y,lambda_,k) * (1 + lambda_ * best.count(1))
    print("\n\nNumber of features selected = {}".format(best.count(1)))
    print("Accuracy = {}%\n\n".format(round(acc*100,2)))