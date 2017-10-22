
### regular and visible players only, pre-determined visibles

from __future__ import division
import random
import pickle
import time
import math
import pandas as pd
import numpy as np
from log import Log
from utils import *
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.time import FollowVisibleActivation
from mesa.time import SimultaneousActivation
from multiprocessing import Pool
from collections import defaultdict
from mesa.datacollection import DataCollector
import itertools

random.seed(0)

class GameAgent(Agent):
    def __init__(self, unique_id, isVisibleNode, isAdversarial, neighbors, visibleColorNodes, model):
        super().__init__(unique_id, model)
        self.game = model
        # whether this node is a visible node
        self.isVisibleNode = isVisibleNode
        # whether this node is an adversarial
        self.isAdversarial = isAdversarial
        self.neighbors = neighbors

        # for each agent initial color is white
        self.new_color = "white"        # state in the next iteration
        self.color = "white"            # current state (other players see this state when updating their states)

        self.visibleColorNodes = visibleColorNodes

        self.unique_id = unique_id

        self.colorChanges = 0

    def __hash__(self):
        return hash(self.unique_id)

    def __eq__(self, other):
        return self.unique_id == other.unique_id

    def instantiateNeighbors(self, model):
        self.neighbors = [agent for agent in model.schedule.agents if
                            agent.unique_id in self.neighbors]

    def instantiateVisibleColorNodes(self, model):
        self.visibleColorNodes = [agent for agent in model.schedule.agents if
                            agent.unique_id in self.visibleColorNodes]


    # determine if there is any visible color node in the neighborhood
    def hasVisibleColorNode(self):
        return len(self.visibleColorNodes) > 0

    # if anybody in the neighbor makes decision
    def hasNeighborDecision(self):
        return [agent.color for agent in self.neighbors if agent.color != "white"]


    def getNeighborMajorColor(self):
        neighbor_color = {"red": 0, "green": 0}
        for a in self.neighbors:
            if a.color != "white":
                neighbor_color[a.color] += 1

        # take one's own decision into account
        if self.color != "white":
            neighbor_color[self.color] += 1

        if neighbor_color["red"] > neighbor_color["green"]:
            # dominant = True if and only if red > green
            dominant = True
            return ("red", dominant)
        elif neighbor_color["red"] < neighbor_color["green"]:
            # dominant = True if and only if red < green
            dominant = True
            return ("green", dominant)
        else:
            # dominant != True if and only if red == green
            dominant = False
            return (random.choice(["red", "green"]), dominant)

    def getNonAdversarialNeighborMajorColor(self):
        neighbor_color = {"red": 0, "green": 0}
        nonAdversarialNeighbors = [neighbor for neighbor in self.neighbors if not neighbor.isAdversarial]   #regular neighbors
        for a in nonAdversarialNeighbors:
            if a.color != "white":
                neighbor_color[a.color] += 1

        # take one's own decision into account
        if self.color != "white":
            neighbor_color[self.color] += 1

        if neighbor_color["red"] > neighbor_color["green"]:
            # dominant = True if and only if red > green
            dominant = True
            return ("red", dominant)
        elif neighbor_color["red"] < neighbor_color["green"]:
            # dominant = True if and only if red < green
            dominant = True
            return ("green", dominant)
        else:
            # dominant != True if and only if red == green
            dominant = False
            return (random.choice(["red", "green"]), dominant)

    # return current majority color
    # this actually corresponds to different players' strategies
    def decision_change(self):
        ###########compute parameters##########
        mid_game = 0
        end_game = 0
        if self.game.time >= 45:
            end_game = 1
        elif self.game.time >= 30 and self.game.time <= 45:
            mid_game = 1
        neighbors = self.degree()

        vis_neighbors = [neighbor for neighbor in self.neighbors if neighbor.isVisibleNode]
        neighbors_vis = float(len(vis_neighbors)) / float(neighbors)
        if len(vis_neighbors) != 0:
            opposite_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(vis_neighbors))
            current_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color != "white" and neighbor.color == self.color])) / float(len(vis_neighbors))
        else:
            opposite_local_vis = 0
            current_local_vis = 0
        inv_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode]
        neighbors_inv = float(len(inv_neighbors)) / float(neighbors)
        if len(inv_neighbors) != 0:
            opposite_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(inv_neighbors))
            current_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color != "white" and neighbor.color == self.color])) / float(len(inv_neighbors))
        else:
            opposite_local_inv = 0
            current_local_inv = 0
        reg_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode and not neighbor.isAdversarial]
        neighbors_reg = float(len(reg_neighbors)) / float(neighbors)
        if len(reg_neighbors) != 0:
            opposite_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(reg_neighbors))
            current_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color != "white" and neighbor.color == self.color])) / float(len(reg_neighbors))
        else:
            opposite_local_reg = 0
            current_local_reg = 0

        ############end computation of parameters###############

        ############begin models################
        if not self.isAdversarial and not self.isVisibleNode:
            # regular node
            if self.hasVisibleColorNode():
                #if regular node has visible neighbors
                y = -3.75 + 1.12 * opposite_local_inv + 1.4 * opposite_local_vis - 0.85 * current_local_inv                 #<-- MODIFY CODE HERE
                prob_of_change = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_change:
                    #change color
                    return "red" if self.color == "green" else "green"
                else:
                    #stay the same color
                    return self.color
            else:
                #if regular node does not have visible neighbors
                y = -3.94 + 0.004 * self.game.time + 2.47 * opposite_local_inv - 0.51 * current_local_inv                   #<-- MODIFY CODE HERE
                prob_of_change = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_change:
                    return "red" if self.color == "green" else "green"
                else:
                    return self.color

        else:
            if self.isVisibleNode:
                #visible node
                if self.hasVisibleColorNode():
                    #if visible node has visible neighbors
                    y = -4.06 + 1.36 * opposite_local_inv + 1.55 * opposite_local_vis - 0.07 * neighbors_inv                #<-- MODIFY CODE HERE
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
                else:
                    #if visible node does not have visible neighbors
                    y = -4.31 + 2.85 * opposite_local_inv                       #<-- MODIFY CODE HERE
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
            else:
                #adversary node
                if self.hasVisibleColorNode():
                    #if adversarial node has visible neighbors
                    y = -3.08 + 0.9 * current_local_vis - 0.15 * neighbors_vis              #<-- MODIFY CODE HERE
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
                else:
                    #if adversarial node does not have visible neighbors
                    y = -2.79 - 1.1 * opposite_local_inv + 1.21 * current_local_inv                 #<-- MODIFY CODE HERE
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color

            ###############end models################

    def decision_choose(self):

        ##########compute parameters############
        mid_game = 0
        end_game = 0
        if self.game.time >= 45:
            end_game = 1
        elif self.game.time >= 30 and self.game.time <= 45:
            mid_game = 1
        neighbors = self.degree()

        vis_neighbors = [neighbor for neighbor in self.neighbors if neighbor.isVisibleNode]
        neighbors_vis = float(len(vis_neighbors)) / float(neighbors)
        if len(vis_neighbors) != 0:
            green_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color == "green"])) / float(len(vis_neighbors))
            red_local_vis = float(len([neighbor for neighbor in vis_neighbors if neighbor.color == "red"])) / float(len(vis_neighbors))
        else:
            green_local_vis = 0
            red_local_vis = 0
        inv_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode]
        neighbors_inv = float(len(inv_neighbors)) / float(neighbors)
        if len(inv_neighbors) != 0:
            green_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color == "green"])) / float(len(inv_neighbors))
            red_local_inv = float(len([neighbor for neighbor in inv_neighbors if neighbor.color == "red"])) / float(len(inv_neighbors))
        else:
            green_local_inv = 0
            red_local_inv = 0
        reg_neighbors = [neighbor for neighbor in self.neighbors if not neighbor.isVisibleNode and not neighbor.isAdversarial]
        neighbors_reg = float(len(reg_neighbors)) / float(neighbors)
        if len(reg_neighbors) != 0:
            green_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color == "green"])) / float(len(reg_neighbors))
            red_local_reg = float(len([neighbor for neighbor in reg_neighbors if neighbor.color == "red"])) / float(len(reg_neighbors))
        else:
            green_local_reg = 0
            red_local_reg = 0

        diff_vis = abs(red_local_vis - green_local_vis)
        diff_inv = abs(red_local_inv - green_local_inv)
        diff_reg = abs(red_local_reg - green_local_reg)

        ##########end computation of parameters##################

        ##############begin models##############
        if self.isAdversarial:
            #adversarial node
            if self.hasVisibleColorNode():
                #if adversarial node has visible neighbors
                y = -2.68 - 0.04 * self.game.time + 1.03 * diff_vis + 1.01 * diff_inv + 0.21 * neighbors_vis    #model for deciding whether to stay or deviate from white   #<-- MODIFY CODE HERE
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding whether to deviate to red or green
                    y2 = -0.37 + 0.83 * green_local_vis
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))          #<-- MODIFY CODE HERE
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
            else:
                #if adversarial node does not have visible neighbors
                y = -2.18 - 0.016 * self.game.time + 1.45 * diff_inv    #model for deciding whether to stay or deviate from white           #<-- MODIFY CODE HERE
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding whether to deviate to red or green
                    y2 = -0.15 + 1.07 * green_local_inv - 0.69 * red_local_inv          #<-- MODIFY CODE HERE
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

        elif self.isVisibleNode:
            #visible node
            if self.hasVisibleColorNode():
                #if visible node has visible neighbors
                y = -1.95 + 0.86 * diff_vis + 0.61 * diff_inv   #model for deciding whether to stay or deviate from white           #<-- MODIFY CODE HERE
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding whether to deviate to red or green
                    y2 = 0.14 - 3.86 * green_local_inv - 1.6 * green_local_vis + 2.63 * red_local_inv + 2.41 * red_local_vis            #<-- MODIFY CODE HERE
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

            else:
                #if visible node does not have visible neighbors
                y = -1.93 + 1.77 * diff_inv #model for deciding whether to stay or deviate from white           #<-- MODIFY CODE HERE
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    #model for deciding whether to deviate to red or green
                    y2 = 0.01 - 4.32 * green_local_inv + 4.32 * red_local_inv                       #<-- MODIFY CODE HERE
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
        else:
            # regular player
            if self.hasVisibleColorNode():
                #if regular node has viisble neighbors
                y = -2.2 - 0.04 * self.game.time + 1.1 * diff_vis + 0.82 * diff_inv + 0.08 * neighbors_vis  #model for deciding whether to stay or deviate from white           #<-- MODIFY CODE HERE
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding whether to deviate to red or green
                    y2 = -0.08 - 2.88 * green_local_inv - 2.07 * green_local_vis + 3.41 * red_local_inv + 1.76 * red_local_vis                  #<-- MODIFY CODE HERE
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
            else:
                #if regular node does not have visible neighbors
                y = -1.94 - 0.03 * self.game.time + 1.63 * diff_inv + 0.01 * neighbors_inv  #model for deciding whether to stay or deviate from white           #<-- MODIFY CODE HERE
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding whether to deviate to red or green
                    y2 = -0.003 - 4.95 * green_local_inv + 5.11 * red_local_inv                 #<-- MODIFY CODE HERE
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

            ###############end models##################

    # function that computes and stores the player's next state in intermediate variable (self.new_color)
    def step(self):
        # check game state
        current_color = getCurrentColor(self.game)
        # if non-adversarial players have reached consensus, set the terminal flag to indicate the game is over
        if current_color["red"] == len(self.game.consensusNodes) or current_color["green"] == len(self.game.consensusNodes):
            self.game.setTerminal()
        else:
            # if non-adversarial players have not reached consensus
            if self.color != "white":
                #if player's current color is not white
                decision_color = self.decision_change() #determine player's color in the next iteration
                if self.color != decision_color:
                    #player has decided to change color
                    self.game.colorChanges += 1     #update the total number of color changes of all players in this simulation
                    if not self.isAdversarial:
                        self.colorChanges += 1      #keep track of the color changes at the player level (non-adversarial players only)
                self.new_color = decision_color     #store the next decision in intermediate variable because player's states are computed sequentially as opposed to parallel: other players need to see this player's current state when they update their states

            else:
                # if player's current color is white
                decision_color = self.decision_choose()     #determine player's color in the next iteration
                if self.color != decision_color:
                    #player has decided to deviate from white
                    self.game.colorChanges += 1
                    if not self.isAdversarial:
                        self.colorChanges += 1
                self.new_color = decision_color

    ## updates the players current color to be the new color
    def advance(self):
        self.color = self.new_color

    def degree(self):
        return len(self.neighbors)


## this function is used to retrieve color information
##  of regular nodes at each time steop
def getCurrentColor(model):
    ret = {"red": 0, "green": 0}
    current_color = [a.color for a in model.schedule.agents\
                if a.unique_id in model.consensusNodes]
    counter = 0
    for item in current_color:
        if item != "white":
            assert item != None     #must either be red or green
            ret[item] += 1
        counter += 1
    return ret


def getRed(model):
    red = 0
    current_color = [a.color for a in model.schedule.agents]
    for color in current_color:
        if color == "red":
            red += 1
    return red


def getGreen(model):
    green = 0
    current_color = [a.color for a in model.schedule.agents]
    for color in current_color:
        if color == "green":
            green += 1
    return green



class DCGame(Model):
    def __init__(self, adjMat, visibles, adversaries):
        self.adjMat = adjMat                            #matrix that keeps track of all players and their neighbors
        self.schedule = SimultaneousActivation(self)    # An activation in which players' states are effectively updated simultaneously as opposed to sequentially
        self.numAgents = len(adjMat)
        self.terminate = False      # if all non-adversarial players have reached consensus, terminal state is achieved
        self.time = 0
        # logging information
        self.log = Log()

        # total number of color changes in a game
        self.colorChanges = 0

        # convert adjMat to adjList
        def getAdjList(adjMat):
            adjList = {key: [] for key in range(self.numAgents)}
            for node in range(self.numAgents):
                #adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == True]
                adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == 'True']
            return adjList

        self.adjList = getAdjList(self.adjMat)

        ############# designate visible, adversarial, and consensus nodes #############
        self.visibleColorNodes = visibles
        self.adversarialNodes = adversaries
        self.consensusNodes = [n for n in range(self.numAgents) if n not in self.adversarialNodes]

        # adversarial nodes and regular nodes should not overlap
        assert set(self.adversarialNodes) & set(self.consensusNodes) == set()
        # visible nodes should belong to regular nodes
        assert set(self.visibleColorNodes) & set(self.consensusNodes) == set(self.visibleColorNodes)

        # logging simulation configuration
        self.log.add("#visible nodes: " + str(self.visibleColorNodes))
        self.log.add("#adversarial nodes: " + str(self.adversarialNodes))
        self.log.add("#consensus nodes: " + str(self.consensusNodes) + '\n')

        ############# initialize all agents #############
        for i in range(self.numAgents):
            # if i is a visible node
            isVisibleNode = i in self.visibleColorNodes
            # if i is an adversarial
            isAdversarial = i in self.adversarialNodes
            # make sure adversarial nodes are not intersected with visible nodes
            assert isVisibleNode & isAdversarial == False

            neighbors = self.adjList[i]

            # visible color nodes in i's neighbors
            vNode = list(set(neighbors) & set(self.visibleColorNodes))

            a = GameAgent(i, isVisibleNode, isAdversarial, neighbors, vNode, self)
            self.schedule.add(a)

        # instantiate all nodes' neighbors and visibleColorNodes
        for agent in self.schedule.agents:
            agent.instantiateNeighbors(self)
            agent.instantiateVisibleColorNodes(self)


        self.datacollector = DataCollector(
                        model_reporters = {"red": getRed, "green": getGreen},
                        agent_reporters = {"agent_color": lambda a: a.color}
                        )

    # simulate the whole model for one step
    def step(self):
        # # # if either red or green reaches consensus, terminates!
        # # in terminal state we do not collect data
        if not self.terminate:
            self.datacollector.collect(self)
            self.schedule.step()
        return self.terminate

    def simulate(self, simulationTimes):
        for i in range(simulationTimes):
            self.updateTime(i)  # update model's time
            terminate = self.step()
            if terminate:
                break

        #added by Yifan
        hasWhitePlayers = False
        if not terminate:
            # if consensus was not reached in the simulation
            for agent in self.schedule.agents:
                if not agent.isAdversarial and not agent.isVisibleNode and agent.color == "white":
                    #at least one consensus player remained white
                    hasWhitePlayers = True

        # output log file to disk
        self.log.outputLog('result/simResult.txt')
        simulatedResult = self.datacollector.get_model_vars_dataframe()
        return (simulatedResult, hasWhitePlayers)

    # update model's clock
    def updateTime(self, t):
        self.time = t

    def setTerminal(self):
        assert self.terminate == False
        self.terminate = True

    def addRecord(self, msg):
        self.log.add(msg)

    # for degub purpose only
    def outputAdjMat(self, path):
        with open(path, 'w') as fid:
            for line in self.adjMat:
                # convert list of boolean values to string values
                tline = ["1" if item else "0" for item in line]
                fid.write(' '.join(tline) + '\n') 



class BatchResult(object):
    def __init__(self, data, dataOnGameLevel, args, networks, numPlayers, numadversaries, numvisibles, hasWhitePlayers, games, labels):
        self.data = data            # self.data records data at each time step
        self.dataOnGameLevel = dataOnGameLevel  # self.dataOnGameLevel records data on each game level
        self.args = args
        self.gameTime = args[1]
        self.numvisibles = numvisibles
        self.numadversaries = numadversaries
        self.networks = networks
        self.consensus_ret = None
        self.dynamics_ret = None
        self.time_ret = None
        self.hasWhitePlayers = hasWhitePlayers
        self.numPlayers = numPlayers
        self.games = games
        self.labels = labels

    # generate a DataFrame where each row corresponds
    # to a simulation
    def generateResult(self):
        consensus_ret = []
        for i in range(len(self.data)):
            if_consensus = 1 if len(self.data[i]) < self.gameTime else 0
            consensus_ret.append((self.games[i], self.labels[i], self.numvisibles[i], self.numadversaries[i],\
                                  self.networks[i], self.numPlayers[i], if_consensus, self.dataOnGameLevel['colorChanges'][i], self.hasWhitePlayers[i]))      #<-- MODIFY CODE HERE
            # consensus_ret.append((self.numVisibleNodes, self.numAdversarialNodes,\
            #                       self.network, self.numPlayers, self.numEdges, if_consensus, self.dataOnGameLevel['hasConflict'][i],
            #                       self.dataOnGameLevel['delay'][i], self.dataOnGameLevel['colorChanges'][i]))
        consensus_ret = pd.DataFrame(consensus_ret)
        self.consensus_ret = consensus_ret

        # generate detailed dynamics for each simulation
        dynamics_ret = {}
        for i in range(len(self.data)):
            dynamics_ret[i] = self.data[i]
        self.dynamics_ret = dynamics_ret

        # generate time to consensus
        time_ret = []
        for i in range(len(self.data)):
            t = len(self.data[i])
            time_ret.append((self.numvisibles[i], self.numadversaries[i], self.networks[i], t))
        time_ret = pd.DataFrame(time_ret)
        self.time_ret = time_ret

    def getConsensusResult(self):
        return self.consensus_ret

    def getDynamicsResult(self):
        return self.dynamics_ret

    def getTimeResult(self):
        return self.time_ret

## extract the {adjMat_num}th adjMat from {YOUR LABEL}_adjacency_matrix.txt
def getAdjMat(adjMat_path, adjMat_num):
    matrices = []
    with open(adjMat_path, "r") as f:
        for row in f:
            row = row.strip('\n')
            row = row.split(' ')
            matrices.append(row)    #extract all adjacency matrices from the file

    adjMat = [] #get the adjacency matrix you want
    occ = [w for w, n in enumerate(matrices) if n[0] == '#']    #find occurrences of '#', which serve as separators of adjMat's in the file
    assert(len(occ) != 0)
    up = occ[adjMat_num-1]   #index of first '#'
    if adjMat_num == 1:
        for x in range(up):
            adjMat.append(matrices[x])  #copy the rows of the {a_adjMat_num}th djacency matrix from the file into adjMat
    else:
        down = occ[adjMat_num-2]
        for x in range(down + 1, up):       #need to specify where to start and stop copying the adjacency matrix within the file
            adjMat.append(matrices[x])

    return adjMat


#define a wrapper function for multi-processing
def simulationFunc(args):
    # dispatch arguments
    numSimulation, gameTime = args

    # inputfile = open('./data_extraction/output/networks/nocomm_noadv_nonzerovis.csv', "r")        #a file containing data and results of simulations conducted on human experiments
    # inputfile = open('./data_extraction/output/networks/nocomm_noadv.csv', "r")
    # inputfile = open('./data_extraction/output/networks/nocomm_testing.csv', "r")
    inputfile = open('./data_extraction/output/networks/nocomm.csv', "r")       # <-- MODIFY CODE HERE

    ###### transform the data in inputfile to a list for processing #########
    listOfSimulations = []
    for line in inputfile:
        line = line.replace('"', '').strip()        #get rid of quotes so that integer strings can be casted to integers
        line = line.split("\t")
        listOfSimulations.append(line)              #append a list of data from each simulation
    listOfSimulations.pop(0)                        #exclude the column titles
    ####### end transforming data in inputfile to a list ##########

    ret = []    # ret contains simulated results
    retOnGameLevel = defaultdict(list)
    networks = []
    numPlayers = []
    numadversaries = []
    numvisibles = []
    hasWhitePlayers = []  #list of whether each game ended with white consensus players or not
    games = []
    labels = []

    ###### extract data from each simulation #########
    for simulation in listOfSimulations:
        adjMat_num = int(simulation[2])     # the correct adjMat to extract from {YOUR LABEL}_adjacency_matrix.txt
        game = int(simulation[2])
        label = simulation[1]
        network = simulation[3]

        visibles = simulation[8]
        # transform a string of visible players into a list
        visibles = visibles.strip('[')
        visibles = visibles.strip(']')
        if visibles == '':
            visibles = []
        else:
            visibles = [int(x) for x in visibles.split(',')]
        adversaries = simulation[6]
        # transform a string of adversarial players into a list
        adversaries = adversaries.strip('[')
        adversaries = adversaries.strip(']')
        if adversaries == '':
            adversaries = []
        else:
            adversaries = [int(x) for x in adversaries.split(',')]
        numvisible = int(simulation[7])
        numadversary = int(simulation[5])
        adjMat_path = 'data_extraction/output/networks/' + label + "_adjacency_matrix.txt"     #path to the file that contains the corresponding adjMat for the simulation
        adjMat = getAdjMat(adjMat_path, adjMat_num)
        ####### end extracting data from each simulation ##########

        print("processing session " + label + " game: " + str(game) + " network: " + network + ", numVisibles: " + str(numvisible) + ", numAdversaries: " + str(numadversary))
        for j in range(numSimulation):
            if j % 100 == 0:
                print("Current number of simulations: ", j)
            model = DCGame(adjMat, visibles, adversaries)            #recreate the setup of the human experiment
            simulatedResult, gameHasWhitePlayer = model.simulate(gameTime)                                              #simulate the experiment using the LogReg models in this code

            #####collect simulation results########
            hasWhitePlayers.append(gameHasWhitePlayer)
            ret.append(simulatedResult)
            networks.append(network)
            numPlayers.append(len(adjMat))
            numadversaries.append(numadversary)
            numvisibles.append(numvisible)
            games.append(game)
            labels.append(label)
            ### a game-level data collector
            retOnGameLevel['colorChanges'].append(model.colorChanges)
            ####end collecting simulation results############

            model.outputAdjMat('result/adjMat.txt')

    # the collected data is actually an object
    result = BatchResult(ret, retOnGameLevel, args, networks, numPlayers, numadversaries, numvisibles, hasWhitePlayers, games, labels)
    return result



def combineResults(result, args, folder=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # result is returned from multi-processing 
    for ret in result:
        ret.generateResult()

    consensus_ret = pd.concat([item.getConsensusResult() for item in result])
    consensus_ret.columns = ['game', 'experiment', '#visibleNodes', '#adversarial', 'network', 'network size', 'ratio',\
                             'colorChanges', 'white nodes']         #<-- MODIFY CODE HERE
    # consensus_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'network size', 'density', 'ratio',\
    #                          'hasConflict', 'delay', 'colorChanges']
    p = os.path.join(folder, 'simulation_results.csv')              #<-- MODIFY CODE HERE (output file that contains results of simulations)
    consensus_ret.to_csv(p, index=None)


if __name__ =="__main__":
    # experimental parameters
    ################################
    numSimulation = 1000                        #<-- MODIFY CODE HERE (number of times each human experiment is simulated)
    gameTime = 60
    ################################

    args = []
    args.append((numSimulation, gameTime))

    # a = list(args[101])
    # a[-2] = 15
    # result = simulationFunc(a)
    # combineResults([result], args, 'result/')
    # a = result.getConsensusResult()
    # a.columns = ['#visibleNodes', '#adversarial', 'network', 'ratio']


    # initialize processes pool
    pool = Pool(processes=40)
    result = pool.map(simulationFunc, args)                 # NEED TO CHANGE: THIS DOESN'T SPEED UP THE CODE SINCE THERE IS ONLY ONE SET OF args (only one process is delegated)
    # combineResults(result, args, 'result/reg_vis_adv_TwoLogReg_LogReg_exp_net/size_20')
    combineResults(result, args, 'result/Logistic_Regression_Models_on_human_experiment_networks')      #<-- MODIFY CODE HERE    (folder where output file is stored in)

    pool.close()
    pool.join()


