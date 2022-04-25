import evolutionaryAlgorithm
import random


def testXorEvoAlgo():
    inputs = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0],[1, 1, 1, 1]]
    #inputs = [[0,0,0],[0,1,1], [1,0,1],[1,1,0]]
    hiddenLayer1Length = 20
    hiddenLayer2Length = 10
    numOutputs = 1

    n_agents = 50

    framework = [len(inputs[0]) - 1, hiddenLayer1Length, hiddenLayer2Length, numOutputs]
    evoAlgo = evolutionaryAlgorithm.EvolutionaryAlgorithm(framework, n_agents)

    num_avg = 100
    while evoAlgo.bestFittness < .95:
        print(evoAlgo.bestFittness, "best overall algo")
        avgChoices = [0] * n_agents
        for i in range(num_avg):
            # generate n agent inputs and recieve the algos decisions, usually this part will be ran multiple times before getting to the completion
            agentInputs = []
            for i in range(n_agents):
                rand = random.randint(0,len(inputs)-1)
                agentInputs.append(inputs[rand])

            choices = evoAlgo.useAgent(agentInputs)

            error = [abs(choices[agent] - agentInputs[agent][3]) for agent in range(n_agents)]
            avgChoices = [(error[agent] + avgChoices[agent]) for agent in range(n_agents)]

        # the percentege that each network gets is its grade
        fittness = [avgChoices[agent] / num_avg for agent in range(n_agents)]

        evoAlgo.makeNewGeneration(fittness)

testXorEvoAlgo()