"""
    Author : Allumolu Harshit
    Email ID : aharshit3@student.nitw.ac.in

    Binary Ant System (BAS) implementation
"""

import random

class Ant:
    """
        Class name : Ant
        Attributes : 
            -- solution : A binary string representing the features selected
                        (1 if selected, 0 otherwise)
            -- fitness : The fitness value obtained using classification accuracy for the features selected by this 
                        particular ant after traversing through the graph
            -- numFeaturesSelected : The number of features selected in the solution
                        which is the number of 1s in the string
    """

    def __init__(self, numFeatures):
        """
            Constructor to initialize an ant and it's attributes
            Arguments : 
                -- numFeatures : The number of features in the dataset
        """

        # A string of length = numFeatures is initialized with all 0s
        self.solution = [0 for _ in range(numFeatures)]
        
        # A variable to store the fitness value obtained from classification
        self.fitness = 0      # initialized with ideal 0 error

        # A variable to track the no. of features selected in the solution
        self.numFeaturesSelected = self.solution.count(1)



class BinaryAntSystem:
    """
        Class name : BinaryAntSystem
        Attributes : 
            -- population : A set of ants
            -- t0 : A list of pheromone values for paths which doesn't select features
            -- t1 : A list of pheromone values for paths which select features
            -- m : number of ants
            -- numFeatures : The number of features in the given dataset
            -- ro : evaporation factor
            -- cf : convergence factor
            -- cfThresholds : threshold values for convergence factor
            -- w : weights for intensification
    """

    def __init__(self, numFeatures):
        """
            Constructor to initialize a binary ant system
            Arguments : 
                -- numFeatures : The number of features in the dataset
        """

        # number of ants    # tuning required
        self.m = 20     # try to come up with an idea to get m from numFeatures

        # initialize t0, t1
        self.population = list()
        self.t0 = list()
        self.t1 = list()
        # pheromone values
        for _ in range(numFeatures):
            self.t0.append(0.5)
            self.t1.append(0.5)

        # numFeatures 
        self.numFeatures = numFeatures

        # evaporation factor
        self.ro = 0.2   # tuning required

        # convergence factor
        self.cf = 0
        # thresholds
        self.cfThresholds = [0.3,0.5,0.7,0.9,0.95]
        # weights for intensification
        self.w = [
            [1,0,0],
            [2/3,1/3,0],
            [1/3,2/3,0],
            [0,1,0],
            [0,0,1]
        ]
    

    def generateNewAnts(self):
        """
            Method name : generateNewAnts
            Arguments : self
            Purpose : 
                -- To create a new generation of ants
        """
        self.population = list()
        # ants
        for _ in range(self.m):
            self.population.append(Ant(self.numFeatures))


    def traverse(self):
        """
            Method name : traverse
            Arguments : self
            Purpose : 
                -- Construct a solution by traversing BAS
        """

        for i in range(self.m):
            while self.population[i].numFeaturesSelected == 0:
                for j in range(self.numFeatures):
                    flag = 0
                    # max probability
                    if self.t0[j] < self.t1[j]:
                        max_p = self.t1[j]
                        flag = 1
                    else:
                        max_p = self.t0[j]
                    # for random paths
                    if random.random() < max_p:
                        self.population[i].solution[j] = flag
                    else:
                        self.population[i].solution[j] = random.randint(0,1)
                # number of features selected
                self.population[i].numFeaturesSelected = self.population[i].solution.count(1)
    

    def convergenceFactor(self):
        """
            Method name : convergenceFactor
            Arguments : self
            Purpose : 
                -- Calculation of convergence factor
        """

        cf = 0
        for i in range(self.numFeatures):
            cf += abs(self.t0[i] - self.t1[i])
        cf /= self.numFeatures
        self.cf = cf


    def updatePheromone(self, Supd):
        """
            Method name : updatePheromone
            Arguments : self
                -- Supd : A set of 3 Ants with iteration best, re-start best and global best solutions
            Purpose : 
                -- Pheromone updation based on constructed solutions
        """

        # evaporation
        for i in range(self.numFeatures):
            self.t0[i] = (1-self.ro) * self.t0[i]
            self.t1[i] = (1-self.ro)* self.t1[i]
        
        # intensification
        ind = int(self.cf/0.2)-1
        for i in range(self.numFeatures):
            temp = 0
            if Supd[0].solution[i] == 1:
                temp += self.w[ind][0]
            if Supd[1].solution[i] == 1:
                temp += self.w[ind][1]
            if Supd[2].solution[i] == 1:
                temp += self.w[ind][2]
            self.t0[i] += self.ro*temp
            self.t1[i] += self.ro*temp

        # check about re-initialization 