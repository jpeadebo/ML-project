import network
import random
import numpy as np


class EvolutionaryAlgorithm:

    def __init__(self, framework, numberAgents):
        self.framework = framework
        self.numberAgents = numberAgents
        self.agents = [network.Network(self.framework) for i in range(self.numberAgents)]
        self.bestAgents = [0]
        self.bestFittness = 0

    def getBestAlgo(self):
        return self.bestAgents[len(self.bestAgents)-1]

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
        maxFittness = fittnessScore.index(max(fittnessScore))
        print(fittnessScore[maxFittness], "max current")
        if self.bestFittness < fittnessScore[maxFittness]:
            self.bestFittness = fittnessScore[maxFittness]
            self.bestAgents.append(self.agents[maxFittness])

        parents = []
        for i in range(int(self.numberAgents / 2)):
            choice = random.uniform(0, totalFittness)
            prev = 0
            for agent in range(len(fittnessScore)):
                pos = fittnessScore[agent] + prev
                if prev <= choice <= pos:
                    parents.append(self.agents[agent])
                    totalFittness -= fittnessScore[agent]
                    fittnessScore[agent] = 0
                    break
                prev = pos
        return parents

    def combineFatherMother(self, father, mother):
        child = []

        dad = father.getWeightMatrix()
        mom = mother.getWeightMatrix()
        for layers in range(len(dad)):
            layer = []
            for node in range(len(dad[layers])):
                pos = [] # im high help
                percentFather = random.randint(0,2)
                percentMother = 2 - percentFather
                for element in range(len(dad[layers][node])):
                    pos.append((dad[layers][node][element] * percentFather) + (mom[layers][node][element] * percentMother) / 2.0)

                layer.append(pos)
            child.append(layer)
        return child

    def makeNewGeneration(self, fittnessScore):
        # pick the new generations parents
        parents = self.pickParents(fittnessScore)
        children = []
        # create n new agents by calculating a new weight matrix given 2 random parents(who can be the same)
        for agents in range(int(self.numberAgents / 2)):

            father = parents[random.randint(0, len(parents) - 1)]
            mother = parents[random.randint(0, len(parents) - 1)]
            child = self.combineFatherMother(father, mother)
            children.append(child)

        # update the agents to be the new children
        for agent in range(len(children)):
            self.agents[agent].setWeightMatrix(children[agent])

        for agent in range(len(children), len(self.agents)):
            self.agents[agent].setWeightMatrix(parents[agent - len(children)].getWeightMatrix())



frame = [3, 2, 2, 1]
EvolutionaryAlgorithm(frame, 5)
