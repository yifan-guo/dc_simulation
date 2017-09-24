
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
from multiprocessing import Pool
from collections import defaultdict
from mesa.datacollection import DataCollector
import itertools

random.seed(0)

class GameAgent(Agent):
    def __init__(self, unique_id, isVisibleNode, isAdversarial, neighbors, visibleColorNodes, inertia, beta, model):
        super().__init__(unique_id, model)
        self.game = model
        # whether this node is a visible node
        self.isVisibleNode = isVisibleNode
        # whether this node is an adversarial
        self.isAdversarial = isAdversarial
        self.neighbors = neighbors

        # for each agent initial color is white
        self.color = "white"

        self.visibleColorNodes = visibleColorNodes

        # probability to make a change
        self.p = inertia
        self.regular_p = 0.84

        # randomize regular players' (excluding visibles)
        # decision
        self.beta = beta

        self.numPlayers = self.game.numAgents

        #added by Yifan
        self.beta_majority = 0.77
        self.beta_tie = 0.9
        self.unique_id = unique_id
        self.threshold = 2

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

    def decision_change(self):
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

        if not self.isAdversarial and not self.isVisibleNode:
            # regular node
            if self.hasVisibleColorNode():
                y = -3.71 - 0.01 * self.game.time + 0.31 * mid_game + 0.56 * end_game - 0.88 * current_local_inv + 1.39 * opposite_local_vis + 1.08 * opposite_local_inv  #trained on all games YG
                prob_of_change = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_change:
                    return "red" if self.color == "green" else "green"
                else:
                    return self.color
            else:
                y = -3.91 - 0.01 * self.game.time + 0.28 * mid_game + 0.56 * end_game - 0.33 * current_local_inv + 2.65 * opposite_local_inv    #trained on all games YG
                prob_of_change = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_change:
                    return "red" if self.color == "green" else "green"
                else:
                    return self.color

        else:
            assert self.isVisibleNode | self.isAdversarial == True
            # pColor, dominant = self.getNeighborMajorColor()

            # if self.isVisibleNode:
            #     return pColor

            # else:
            #     return "red" if pColor == "green" else "green"

            if self.isVisibleNode:
                #visible node
                # y = -4.04  - 0.41 * current_local_vis + 1.89 * opposite_local_reg + 0.93 * opposite_local_vis + 0.18 * neighbors_vis - 0.05  * neighbors_reg
                if self.hasVisibleColorNode():
                    y = -3.83 + 0.23 * end_game + 1.55 * opposite_local_vis + 1.03 * opposite_local_inv - 0.06 * neighbors_inv - 0.38 * current_local_inv  # trained on all games YG
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
                else:
                    y = -4.36 + 0.22 * end_game + 2.84 * opposite_local_inv     #trained on all games YG
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
            else:
                #adversary node
                # opposite_local = float(len([neighbor for neighbor in self.neighbors if neighbor.color != "white" and neighbor.color != self.color])) / float(len(self.neighbors))
                # y = -1.72 + 0.35 * mid_game + 0.27 * end_game + 0.02 * neighbors - 2.68  * opposite_local
                if self.hasVisibleColorNode():
                    y = -2.94 - 0.01 * self.game.time + 0.49 * end_game - 0.56 * opposite_local_inv - 0.20 * neighbors_vis + 0.69 * current_local_vis + 1 * current_local_inv       #trained on all games YG
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color
                else:
                    y = -2.72 - 0.01 * self.game.time + 0.26 * mid_game + 0.56 * end_game -1.03 * opposite_local_inv + 1.29 * current_local_inv     # trained on all games YG
                    prob_of_change = float(1) / float(1 + math.exp(-y))
                    if random.random() < prob_of_change:
                        return "red" if self.color == "green" else "green"
                    else:
                        return self.color

    def decision_choose(self):

        mid_game = 0
        end_game = 0
        if self.game.time >= 45:
            end_game = 1
        elif self.game.time >= 30 and self.game.time <= 45:
            mid_game = 1
        neighbors = self.degree()

        # red_local = float((len([neighbor for neighbor in self.neighbors if neighbor.color == "red"]))) / float(len(self.neighbors))
        # green_local = float((len([neighbor for neighbor in self.neighbors if neighbor.color == "green"]))) / float(len(self.neighbors))

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

        diff_vis = abs(red_local_vis - green_local_vis)
        diff_inv = abs(red_local_inv - green_local_inv)

        if self.isAdversarial:
            # red_global = float(len([n for n in self.game.schedule.agents if n.color == "red"]))/float(len(self.game.adjMat))
            # green_global = float(len([n for n in self.game.schedule.agents if n.color == "green"]))/float(len(self.game.adjMat))
            # y = -2.03 - 0.07 * self.game.time + 0.9 * red_local + 0.71 * green_local + 1.01 * red_global + 1.37* green_global
            if self.hasVisibleColorNode():
                # y = -2.35  - 0.04 * self.game.time + 0.8 * diff_reg + 1 * diff_vis
                y = -2.68 + 1.01 * diff_inv + 1.03 * diff_vis - 0.04 * self.game.time + 0.21 * neighbors_vis    # trained on all games YG
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding which color
                    # y2 = 0.08 - 4.6 * green_local + 4.68 * red_local
                    # y2 = 0.01 + 0.68 * green_local - 0.87 * red_local
                    y2 = -0.34 + 0.8 * green_local_vis - 0.07 * red_local_vis    #trained on all games YG
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
            else:
                # y = -2.21 + 1.16 * diff_reg
                y = -2.23 - 1.28 * mid_game + 1.34 * diff_inv   # trained on all games YG
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding which color
                    # y2 = 0.08 - 4.6 * green_local + 4.68 * red_local
                    y2 = -0.15 + 1.07 * green_local_inv - 0.69 * red_local_inv      #trained on all games YG
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
        elif self.isVisibleNode:
            # red_global = float(len([n for n in self.game.schedule.agents if n.color == "red"]))/float(len(self.game.adjMat))
            # green_global = float(len([n for n in self.game.schedule.agents if n.color == "green"]))/float(len(self.game.adjMat))
            # y = -2.03 - 0.07 * self.game.time + 0.9 * red_local + 0.71 * green_local + 1.01 * red_global + 1.37* green_global
            # y = -1.86 + 0.51 * red_local_reg + 1.62 * green_local_reg
            # y = -1.79 - 0.04 * self.game.time + 1.63 * majority_local_reg
            # y = -1.78 + 1.06 * diff_reg
            if self.hasVisibleColorNode():
                y = -1.95  + 0.61 * diff_inv + 0.86 * diff_vis    #trained on all games YG
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding which color
                    # y2 = 0.08 - 4.6 * green_local + 4.68 * red_local
                    # y2 = 0.14 - 4.44 * green_local_reg + 4.96 * red_local_reg
                    y2 = 0.14 - 3.86 * green_local_inv - 1.61 * green_local_vis + 2.63 * red_local_inv + 2.41 * red_local_vis   #trained on all games YG
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"

            else:
                y = -1.95 - 3.38 * mid_game + 1.96 * diff_inv   #trained on all games YG
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding which color
                    # y2 = 0.08 - 4.6 * green_local + 4.68 * red_local
                    # y2 = 0.14 - 4.44 * green_local_reg + 4.96 * red_local_reg
                    y2 = 0.01 - 4.32 * green_local_inv + 4.32 * red_local_inv  # trained on all games YG
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
        else:
            # regular player
            # red_global = float(len([n for n in self.game.schedule.agents if n.color == "red"]))/float(len(self.game.adjMat))
            # green_global = float(len([n for n in self.game.schedule.agents if n.color == "green"]))/float(len(self.game.adjMat))
            # y = -2.03 - 0.07 * self.game.time + 0.9 * red_local + 0.71 * green_local + 1.01 * red_global + 1.37* green_global
            if self.hasVisibleColorNode():
                # y = -2.12 - 0.05 * self.game.time + 0.83 * red_local_reg + 0.89 * red_local_vis + 1.17 * green_local_reg + 0.84 * green_local_vis #trained on games with only regular and visibles
                # y = -2.09 - 0.03 * self.game.time + 0.9 * diff_reg + 0.9 * diff_vis
                y = -2.21 - 0.04 * self.game.time - 0.67 * mid_game + 0.08 * neighbors_vis + 0.80 * diff_inv + 1.09 * diff_vis   # trained on all games YG
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding which color
                    # y2 = 0.08 - 4.6 * green_local + 4.68 * red_local             
                    # y2 = 0.02 - 3.48 * green_local_reg - 1.95 * green_local_vis + 3.48 * red_local_reg + 2.12 * red_local_vis      #trained on games with only regular and visible players
                    y2 =  -0.08 - 2.88 * green_local_inv - 2.07 * green_local_vis + 3.41 * red_local_inv + 1.76 * red_local_vis     #trained on all games YG
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"
            else:
                # y = -1.96 - 0.05 * self.game.time + 1.52 * red_local + 1.53 * green_local     #trained on games with only regular and visibles
                # y = -1.92 + 1.72 * diff_reg - 0.04 * self.game.time
                y = -1.96 - 0.02 * self.game.time - 0.91 * mid_game + 1.58 * diff_inv + 0.01 * neighbors_inv  #trained on all games YG
                prob_of_choose = float(1) / float(1 + math.exp(-y))
                if random.random() < prob_of_choose:
                    # model for deciding which color
                    # y2 = 0.08 - 4.6 * green_local + 4.68 * red_local              #trained on games with only regular and visibles
                    y2 = -4.94 * green_local_inv + 5.14 * red_local_inv - 0.93 * mid_game     #trained on all games YG
                    prob_of_choose_color = float(1) / float(1 + math.exp(-y2))
                    if random.random() < prob_of_choose_color:
                        return "red"
                    else:
                        return "green"
                else:
                    return "white"


    # make a decision
    def step(self, order):
        # check game state
        current_color = getCurrentColor(self.game)
        # print("red: " + str(current_color["red"]) + ", green: " + str(current_color["green"]) + ", goal: " + str(self.numPlayers - self.game.numAdversarialNodes))
        if current_color["red"] == (self.numPlayers - self.game.numAdversarialNodes) or current_color["green"] == (self.numPlayers - self.game.numAdversarialNodes):
            self.game.setTerminal()
        else:
            #no consensus yet
            if self.color != "white":
                #if node already picked a color
                decision_color = self.decision_change()
                self.color = decision_color
                self.game.colorChanges += 1   

            else:
                # node has not yet picked a color
                decision_color = self.decision_choose()
                self.color = decision_color
                self.game.colorChanges += 1  


    def degree(self):
        return len(self.neighbors)


## this function is used to retrieve color information
##  of regular nodes at each time steop
def getCurrentColor(model):
    ret = {"red": 0, "green": 0}
    current_color = [a.color for a in model.schedule.agents\
                if a.unique_id in model.regularNodes]

    # a  = set(model.visibleColorNodes) & set(model.regularNodes) == set(model.visibleColorNodes)
    # print(a)
    counter = 0
    for item in current_color:
        if item != "white":
            if item == None:
                print("unique id: " + str(model.schedule.agents[counter].unique_id))
            ret[item] += 1
        counter += 1
    return ret


# get the number of nodes selecting red in each time step
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
    def __init__(self, adjMat, G, numVisibleColorNodes, numAdversarialNodes, inertia, beta, delay, visibles, adversaries):
        self.adjMat = adjMat
        self.numVisibleColorNodes = numVisibleColorNodes
        self.numAdversarialNodes = numAdversarialNodes
        # self.adversarialNodes = []
        self.visibleColorNodes = []
        self.regularNodes = []
        self.schedule = FollowVisibleActivation(self)
        self.numAgents = len(adjMat)
        self.inertia = inertia
        # if there are 20 consensus colors then a
        # terminal state is reached
        self.terminate = False
        self.time = 0
        # logging information
        self.log = Log()

        ##  temporarily added this for figuring out 
        ##  why visible nodes have no help
        self.hasConflict = False

        # randomize regular players (exclude visibles)
        # decision
        self.beta = beta

        # a amount of time to delay ordinary players' decision
        # ordinary players = players who are neither visibles
        # nor has any visibles in their neighbor
        self.delay = delay

        # total number of color changes in a game
        self.colorChanges = 0

        # addded by Yifan
        self.reach_of_adversaries = 0
        self.reach_of_visibles = 0
        self.total_cnt_of_adversaries = 0
        self.total_cnt_of_visibles = 0
        self.graph = G

        # convert adjMat to adjList
        def getAdjList(adjMat):
            adjList = {key: [] for key in range(self.numAgents)}
            for node in range(self.numAgents):
                #adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == True]
                adjList[node] = [idx for idx, value in enumerate(adjMat[node]) if value == 'True']
            return adjList

        self.adjList = getAdjList(self.adjMat)
        #return the subset of L availableNodes in G with the largest number of distinct neighbors
        def getSubsetWithMaxDistinctNeighbors(availableNodes, G, L):
            acc = []
            max_cnt = 0
            local_cnt = 0
            hasBeenConsidered = [False for i in range(self.numAgents)]
            graph = nx.convert.to_dict_of_lists(G)
            for subset in itertools.combinations(availableNodes, L):
                upper_bound = 0
                for agent in subset:
                    upper_bound += len(graph[agent])
                if upper_bound < max_cnt:
                    continue
                # compute reach
                for agent in subset:
                    for neighbor in G.neighbors(agent):
                        if neighbor not in subset and hasBeenConsidered[neighbor] == False:
                            local_cnt += 1
                            hasBeenConsidered[neighbor] = True
                if local_cnt > max_cnt:
                    max_cnt = local_cnt
                    acc.clear()
                    for agent in subset:
                        acc.append(agent)
                local_cnt = 0
                hasBeenConsidered = [False for i in range(self.numAgents)]
            return acc

        ############# designate visible #############
        # node_deg = [(idx, count(adjMat[idx])) for idx in range(self.numAgents)]
        # availableNodes = [item[0] for item in node_deg]
        # random.shuffle(availableNodes)
        # availableNodes.sort(key=lambda x : x)
        # self.visibleColorNodes = getSubsetWithMaxDistinctNeighbors(availableNodes, G, numVisibleColorNodes)
        # self.visibleColorNodes = [item for item in availableNodes[:self.numVisibleColorNodes]]
        self.visibleColorNodes = visibles
        # for visibleNode in self.visibleColorNodes:
        #     availableNodes.remove(visibleNode)


        ############# designate adversarial ###############
        # self.adversarialNodes = getSubsetWithMaxDistinctNeighbors(availableNodes, G, numAdversarialNodes)
        # self.adversarialNodes = [item for item in availableNodes[:self.numAdversarialNodes]]
        self.adversarialNodes = adversaries


        # ================ prev version: designate adversarial and visible nodes ===========
        # node_deg = [(idx, count(adjMat[idx])) for idx in range(self.numAgents)]
        # all_nodes = [item[0] for item in node_deg]
        # random.shuffle(node_deg)
        # self.adversarialNodes = [item[0] for item in node_deg[:self.numAdversarialNodes]]

        # reach_of_adversaries = 0
        # total_cnt_of_adversaries = 0
        # hasBeenReached = dict.fromkeys(all_nodes, False)                
        # for adversarialNode in self.adversarialNodes:
        #     for neighbor in G.neighbors(adversarialNode):
        #         if neighbor not in self.adversarialNodes:
        #             total_cnt_of_adversaries += 1
        #         if neighbor not in self.adversarialNodes and hasBeenReached[neighbor] == False:
        #             reach_of_adversaries += 1
        #             hasBeenReached[neighbor] = True
        # self.reach_of_adversaries = reach_of_adversaries
        # self.total_cnt_of_adversaries = total_cnt_of_adversaries

        # ############# designate visible nodes #############
        # availableNodes = shuffled(node_deg[self.numAdversarialNodes:])
        # self.visibleColorNodes = [item[0] for item in availableNodes[:self.numVisibleColorNodes]]

        # reach_of_visibles = 0
        # total_cnt_of_visibles = 0
        # hasBeenReached = dict.fromkeys(all_nodes, False)
        # for visibleColorNode in self.visibleColorNodes:
        #     for neighbor in G.neighbors(visibleColorNode):
        #         if neighbor not in self.adversarialNodes and neighbor not in self.visibleColorNodes:
        #             total_cnt_of_visibles += 1
        #         if neighbor not in self.adversarialNodes and neighbor not in self.visibleColorNodes and hasBeenReached[neighbor] == False:
        #             reach_of_visibles += 1
        #             hasBeenReached[neighbor] = True
        # self.reach_of_visibles = reach_of_visibles
        # self.total_cnt_of_visibles = total_cnt_of_visibles

        # ===============================

        self.regularNodes = [n for n in range(self.numAgents) if n not in self.adversarialNodes]
        # make sure we have 20 regular nodes
        # assert len(self.regularNodes) ==20

        assert set(self.adversarialNodes) & set(self.visibleColorNodes) == set()
        # adversarial nodes and regular nodes should not overlap
        assert set(self.adversarialNodes) & set(self.regularNodes) == set()
        # visible nodes should belong to regular nodes
        assert set(self.visibleColorNodes) & set(self.regularNodes) == set(self.visibleColorNodes)

        # logging simulation configuration
        self.log.add("#visible nodes: " + str(self.visibleColorNodes))
        self.log.add("#adversarial nodes: " + str(self.adversarialNodes))
        self.log.add("#regular nodes: " + str(self.regularNodes) + '\n')

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
            
            inertia = self.inertia
            beta = self.beta

            # print("Add agent:", (i, visibleNode, adversarial, neighbors, visibleColorNodes))
            a = GameAgent(i, isVisibleNode, isAdversarial, neighbors, vNode, inertia, beta, self)
            self.schedule.add(a)

        # instantiate all nodes' neighbors and visibleColorNodes
        for agent in self.schedule.agents:
            agent.instantiateNeighbors(self)
            agent.instantiateVisibleColorNodes(self)


        self.datacollector = DataCollector(
                        model_reporters = {"red": getRed, "green": getGreen},
                        agent_reporters = {"agent_color": lambda a: a.color}
                        )

    def getReachOfAdversaries(self):
        return self.reach_of_adversaries

    def getReachOfVisibles(self):
        return self.reach_of_visibles

    def getTotalCntOfAdversaries(self):
        return self.total_cnt_of_adversaries

    def getTotalCntOfVisibles(self):
        return self.total_cnt_of_visibles

    # simulate the whole model for one step
    def step(self):
        # # # if either red or green reaches consensus, terminates!
        # # in terminal state we do not collect data
        if not self.terminate:
            self.datacollector.collect(self)
            self.schedule.step(self.delay)
        return self.terminate

    def simulate(self, simulationTimes):
        for i in range(simulationTimes):
            # update model's time
            # print("simulation time: " + str(i))
            self.updateTime(i)
            terminate = self.step()
            if terminate:
                break

        #added by Yifan
        isRegWhite = False
        # output log file to disk
        if not terminate:
            # did not reach consensus
            for agent in self.schedule.agents:
                if not agent.isAdversarial and not agent.isVisibleNode and agent.color == "white":
                    #at least one regular player remained white
                    isRegWhite = True


        self.log.outputLog('result/simResult.txt')
        simulatedResult = self.datacollector.get_model_vars_dataframe()
        # print(simulatedResult)
        return (simulatedResult, isRegWhite)

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
    def __init__(self, data, dataOnGameLevel, args, arg_id, networks, numPlayers, numadversaries, numvisibles, isGameWhite, games, labels):
        # used to uniquely pair BatchResult and args
        self.ret_id = arg_id
        # self.data records data at each time step
        self.data = data

        # self.dataOnGameLevel records data on 
        # each game level
        self.dataOnGameLevel = dataOnGameLevel
        ###

        self.args = args
        self.gameTime = args[1]
        # self.numVisibleNodes = args[3]
        # self.numVisibleNodes = args[2]
        self.numvisibles = numvisibles
        # self.numAdversarialNodes = args[4]
        self.numadversaries = numadversaries
        # self.network = args[5]
        self.networks = networks
        self.consensus_ret = None
        self.dynamics_ret = None
        self.time_ret = None

        self.isGameWhite = isGameWhite

        #added by Yifan
        # self.reach_of_adversaries = reach_of_adversaries
        # self.reach_of_visibles = reach_of_visibles
        # self.total_cnt_of_adversaries = total_cnt_of_adversaries
        # self.total_cnt_of_visibles = total_cnt_of_visibles
        # self.numPlayers = args[2] + args[4]
        self.numPlayers = numPlayers
        # self.numEdges = args[6]

        self.games = games
        self.labels = labels

    def generateResult(self):
        # generate a DataFrame where each row corresponds
        # to a simulation
        consensus_ret = []
        for i in range(len(self.data)):
            if_consensus = 1 if len(self.data[i]) < self.gameTime else 0
            consensus_ret.append((self.games[i], self.labels[i], self.numvisibles[i], self.numadversaries[i],\
                                  self.networks[i], self.numPlayers[i], if_consensus, self.dataOnGameLevel['hasConflict'][i],
                                  self.dataOnGameLevel['delay'][i], self.dataOnGameLevel['colorChanges'][i], self.isGameWhite[i]))
            # consensus_ret.append((self.numVisibleNodes, self.numAdversarialNodes,\
            #                       self.network, if_consensus, self.dataOnGameLevel['hasConflict'][i],
            #                       self.dataOnGameLevel['delay'][i], self.dataOnGameLevel['colorChanges'][i], self.reach_of_adversaries[i], self.reach_of_visibles[i], self.total_cnt_of_adversaries[i], self.total_cnt_of_visibles[i]))
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


# def getAdjMat(net, numPlayers, numRegularPlayers, numAdversarialNodes):
#     # network parameters
#     ################################
#     ### CODE FROM Zlatko ###
#     # each new node is connected to m new nodes
#     m = 3
#     no_consensus_nodes_range = range(11)
#     # max degrees
#     maxDegree = 17
#     BA_edges = [(numRegularPlayers + no_consensus_nodes - 3) * m for \
#                             no_consensus_nodes in no_consensus_nodes_range]
#     ERD_edges = [edges_no for edges_no in BA_edges]
#     ERS_edges = [int(math.ceil(edges_no/2.0)) for edges_no in ERD_edges]
#     ################################ 

#     # generate adjMat according to network type
#     if net == 'Erdos-Renyi-dense':
#         adjMat, G = ErdosRenyi(numPlayers, ERD_edges[numAdversarialNodes], maxDegree)
#     elif net == 'Erdos-Renyi-sparse':
#         adjMat, G = ErdosRenyi(numPlayers, ERS_edges[numAdversarialNodes], maxDegree)
#     else:
#         adjMat, G = AlbertBarabasi(numPlayers, m, maxDegree)

#     return (adjMat, G)

# def getAdjMat(net, numPlayers, numEdges):
#     m = 3
#     maxDegree = 17
#     adjMat, G = ErdosRenyi(numPlayers, numEdges, maxDegree)
#     return (adjMat, G)

def getAdjMat(network_path, i):
    matrices = []   #all matrices for this game
    with open(network_path, "r") as f:
        for row in f:
            row = row.strip('\n')
            row = row.split(' ')
            matrices.append(row)

    adjMat = [] #get the one you want
    occ = [w for w, n in enumerate(matrices) if n[0] == '#']
    assert(len(occ) != 0)
    up = occ[i-1]   #index of first tag
    if i == 1:
        for x in range(up):
            adjMat.append(matrices[x])
    else:
        down = occ[i-2]
        for x in range(down + 1, up):
            adjMat.append(matrices[x])

    fladjMat = [[0 for n in range(len(adjMat))] for k in range(len(adjMat))]  #create a binary eq of adjMat
    for a in range(len(adjMat)):
        for b in range(len(adjMat)):
            if adjMat[a][b] == 'True':
                fladjMat[a][b] = 1.
            else:
                fladjMat[a][b] = 0.

    numadjMat = np.matrix(fladjMat)
    G =nx.from_numpy_matrix(numadjMat)  #reconstruct the graph

    return (adjMat, G)


#define a wrapper function for multi-processing
def simulationFunc(args):
    # dispatch arguments
    # numSimulation, gameTime, numRegularPlayers, numVisibleNodes, \
    #             numAdversarialNodes, net, inertia, beta, delay, arg_id = args
    numSimulation, gameTime, numVisibleNodes, \
                numAdversarialNodes, inertia, beta, delay, arg_id = args

    # calculate how many players we have
    # numPlayers = numRegularPlayers + numAdversarialNodes

    inputfile = open('./data_extraction/output/networks/nocomm.csv', "r")
    content = []        #list of games
    for line in inputfile:
        line = line.replace('"', '').strip()
        line = line.split("\t")
        content.append(line)
    content.pop(0)

    # ret contains simulated results
    ret = []
    # reaches_adversaries = []
    # reaches_visibles = []
    # total_cnt_of_adversaries = []
    # total_cnt_of_visibles = []
    retOnGameLevel = defaultdict(list)
    networks = []
    numPlayers = []
    numadversaries = []
    numvisibles = []
    isGameWhite = []  #number of games in which one regular player is still white
    games = []
    labels = []
    for line in content:
        #each game
        network_num = int(line[2])
        game = int(line[2])
        label = line[1]
        network = line[3]

        visibles = line[8]
        visibles = visibles.strip('[')
        visibles = visibles.strip(']')
        if visibles == '':
            visibles = []
        else:
            visibles = [int(x) for x in visibles.split(',')]
        print(visibles)
        adversaries = line[6]
        adversaries = adversaries.strip('[')
        adversaries = adversaries.strip(']')
        if adversaries == '':
            adversaries = []
        else:
            adversaries = [int(x) for x in adversaries.split(',')]
        print(adversaries)
        numvisible = int(line[7])
        numadversary = int(line[5])
        network_path = 'data_extraction/output/networks/' + label + "_adjacency_matrix.txt"
        adjMat, G = getAdjMat(network_path, network_num)
        print("processing session " + label + " " + network + ", numVisibles: " + str(numvisible) + ", numAdversaries: " + str(numadversary))
        for j in range(numSimulation):
            if j % 100 == 0:
                print("Current number of simulations: ", j)
            # adjMat, G = getAdjMat(net, numPlayers, numRegularPlayers, numAdversarialNodes)
            # adjMat, G = getAdjMat(net, numPlayers, numEdges)
            model = DCGame(adjMat, G, numvisible, numadversary, inertia, beta, delay, visibles, adversaries)
            simulatedResult, isRegWhite = model.simulate(gameTime)
            if isRegWhite:
                isGameWhite.append(True)
            else:
                isGameWhite.append(False)
            ret.append(simulatedResult)
            networks.append(network)
            numPlayers.append(len(adjMat))
            numadversaries.append(numadversary)
            numvisibles.append(numvisible)
            games.append(game)
            labels.append(label)
            # reaches_adversaries.append(model.getReachOfAdversaries())
            # reaches_visibles.append(model.getReachOfVisibles())
            # total_cnt_of_adversaries.append(model.getTotalCntOfAdversaries())
            # total_cnt_of_visibles.append(model.getTotalCntOfVisibles())
            ### a game-level data collector
            retOnGameLevel['hasConflict'].append(model.hasConflict)
            retOnGameLevel['delay'].append(model.delay)
            retOnGameLevel['colorChanges'].append(model.colorChanges)
            ###

            # print(simulatedResult)
            model.outputAdjMat('result/adjMat.txt')

    # the collected data is actually an object
    #result = BatchResult(ret, reaches_adversaries, reaches_visibles, total_cnt_of_adversaries, total_cnt_of_visibles, retOnGameLevel, args, arg_id)
    result = BatchResult(ret, retOnGameLevel, args, arg_id, networks, numPlayers, numadversaries, numvisibles, isGameWhite, games, labels)
    return result



def combineResults(result, args, folder=None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    inertia = args[0][-4]
    beta = args[0][-3]

    # result is returned from multi-processing 
    for ret in result:
        ret.generateResult()

    consensus_ret = pd.concat([item.getConsensusResult() for item in result])
    consensus_ret.columns = ['game', 'experiment', '#visibleNodes', '#adversarial', 'network', 'network size', 'ratio',\
                             'hasConflict', 'delay', 'colorChanges', 'white nodes']
    # consensus_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'ratio',\
    #                          'hasConflict', 'delay', 'colorChanges', 'reach_of_adversaries', 'reach_of_visibles', 'total_cnt_of_adversaries','total_cnt_of_visibles']
    # consensus_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'network size', 'density', 'ratio',\
    #                          'hasConflict', 'delay', 'colorChanges']
    p = os.path.join(folder, 'consensus_inertia=%.2f_beta=%.2f.csv' % (inertia, beta))
    consensus_ret.to_csv(p, index=None)


    # time_ret = pd.concat([item.getTimeResult() for item in result])
    # time_ret.columns = ['#visibleNodes', '#adversarial', 'network', 'time']
    # p = os.path.join(folder, 'time_inertia=%.2f.csv' % inertia)
    # time_ret.to_csv(p, index=None)

    # dynamics_ret = {args[idx]: item.getDynamicsResult() for idx, item in enumerate(result)}
    # p = os.path.join(folder, 'dynamics_inertia=%.2f.p' % inertia)
    # with open(p, 'wb') as fid:
    #     pickle.dump(dynamics_ret, fid)



if __name__ =="__main__":
    # iterate over all inertia values
    for inertia in np.linspace(0.87, 0.87, 1):
        print("Current inertia: ", inertia)
        for beta in np.linspace(1.0, 1.0, 1):

            # experimental parameters
            ################################
            numSimulation = 100
            gameTime = 60
            # inertia = 0.5
            numRegularPlayers = 20
            # numRegularPlayersList = [17 + (n * 3) for n in range(5)]
            ################################

            args = []
            networks = ['Erdos-Renyi-dense', 'Erdos-Renyi-sparse', 'Barabasi-Albert']
            # networks = ['Erdos-Renyi']
            numVisibleNodes_ = [0]
            numAdversarialNodes_ = [0]
            delayTime_ = [0]
            # ER_edges = [23, 26, 45, 51]
            ER_edges = [25 + 5 * i for i in range(15)]

            # get all combinations of parameters
            counter = 0
            # for net in networks:
            for numVisible in numVisibleNodes_:
                for numAdv in numAdversarialNodes_:
                    for delay in delayTime_:
                        # for numRegularPlayers in numRegularPlayersList:
                        args.append((numSimulation, gameTime, numVisible,
                                                 numAdv, inertia, beta, delay, counter))
                        counter += 1
                        # for numEdges in ER_edges:
                        #     print("Generate parameters combinations: ", (net, numVisible, numAdv))
                        #     args.append((numSimulation, gameTime, numRegularPlayers, numVisible,
                        #                      numAdv, net, numEdges, inertia, beta, delay, counter))
                        #     counter += 1

            # a = list(args[101])
            # a[-2] = 15
            # result = simulationFunc(a)
            # combineResults([result], args, 'result/')
            # a = result.getConsensusResult()
            # a.columns = ['#visibleNodes', '#adversarial', 'network', 'ratio']


            # initialize processes pool
            pool = Pool(processes=40)
            result = pool.map(simulationFunc, args)
            combineResults(result, args, 'result/reg_vis_adv_TwoLogReg_LogReg_exp_net_faster/size_20')

            pool.close()
            pool.join()


