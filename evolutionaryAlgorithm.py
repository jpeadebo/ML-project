import network
import random
import numpy as np


class EvolutionaryAlgorithm:

    def __init__(self, framework, numberAgents):
        self.framework = framework
        self.numberAgents = numberAgents
        self.agents = [network.Network(self.framework)] * self.numberAgents

    # takes in inputs and outputs the decisions made by each agent
    def useAgent(self, inputs):
        agentChoices = []
        for agent in range(len(self.agents)):
            self.agents[agent].setInputs(inputs[agent])
            agentChoices.append(self.agents[agent].feedForward())
        return agentChoices

    # higher fittness scores are more likely to be picked as the take up a longer range
    def pickParents(self, fittnessScore):
        totalFittness = sum(fittnessScore)
        parents = []
        for i in range(self.numberAgents / 2):
            choice = random.uniform(0, totalFittness)
            prev = 0
            for agent in range(len(fittnessScore)):
                pos = fittnessScore[agent] + prev
                if prev < choice <= pos:
                    parents.append(self.agents[agent])
                    totalFittness -= fittnessScore[agent]
                    fittnessScore[agent] = 0
                    break
                prev = pos
        return parents

    def combineFatherMother(self, father, mother):
        child = [np.random.randint(1, 1, size=(self.framework[i], self.framework[i - 1])) for i in
                 range(1, len(self.framework))]

        dad = self.agents[father].getWeightMatrix()
        mom = self.agents[mother].getWeightMatrix()
        for layers in range(len(dad)):
            for node in range(len(dad[layers])):
                percentFather = random.uniform(0, 2)
                percentMother = 2 - percentFather
                for element in range(len(dad[layers][node])):
                    child[layers][node][element] = ((dad[layers][node][element] * percentFather) + (mom[layers][node][element] * percentMother)) / 2.0
        return child

    def makeNewGeneration(self, fittnessScore):
        # pick the new generations parents
        parents = self.pickParents(fittnessScore)
        children = []
        # create n new agents by calculating a new weight matrix given 2 random parents(who can be the same)
        for agents in range(self.numberAgents):
            father = random.randint(0, len(parents) - 1)
            mother = random.randint(0, len(parents) - 1)
            child = self.combineFatherMother(father, mother)
            children.append(child)

        # update the agents to be the new children
        for agent in range(len(self.agents)):
            self.agents[agent].setWeightMatrix(children[agent])


frame = [3, 2, 2, 1]
EvolutionaryAlgorithm(frame, 5)
