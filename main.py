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


def MOFS-BACO(numFeatures):
    """
        Function name : MOFS-BACO
        Arguments : 
            --
    """
    
    # intialization
    graph = BinaryAntSystem(numFeatures)
    P = 10              # OPS interval
    iterations = 3000   # tuning required
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
        
        """
            solution evaluation code block
            use k-nn with k=1, tuning required
        """

        best = Ant(numFeatures)
        for i in range(graph.m):
            # error less than best => make it best
            if graph.population[i].error < best.error:
                best = graph.population[i]         
            # error == best => make it best if less number of features are selected
            elif graph.population[i].error == best.error:
                if graph.population[i].numFeaturesSelected < best.numFeaturesSelected:
                    best = graph.population[i]
        
        Supd[0] = best
        if best.error < Supd[1].error:
            Supd[1] = best
        if best.error < Supd[2].error:
            Supd[2] = best
        
        # calculate convergence factor
        graph.convergenceFactor()   # graph.cf

        # update pheromone values
        graph.updatePheromone(Supd)