import math
import os
import random
import networkx as nx
from copy import deepcopy
import pandas as pd

if __name__ == "__main__":
	consensus_ret = []
	m = 3
	maxDegree = 17
	d = maxDegree
	no_consensus_nodes_range = range(11)
	numRegularPlayers = 20
	numAdversarialNodes = [0, 1, 2, 3, 4, 5]
	numVisibleColorNodes = [0, 1, 2, 3, 4, 5]
	networks = ['Barabasi-Albert', 'Erdos-Renyi-dense', 'Erdos-Renyi-sparse']
	for i in range(100):
		for net in networks:
			print('network: ' + net)
			for numAdversaries in numAdversarialNodes:
				print('numAdversaries: ' + str(numAdversaries))
				for numVisibles in numVisibleColorNodes:
					print('numVisibles: ' + str(numVisibles))
					BA_edges = [(numRegularPlayers + no_consensus_nodes - 3) * m for no_consensus_nodes in no_consensus_nodes_range]
					ERD_edges = [edges_no for edges_no in BA_edges]
					ERS_edges = [int(math.ceil(edges_no/2.0)) for edges_no in ERD_edges]
					n = numRegularPlayers + numAdversaries
					while True:
						if net == 'Erdos-Renyi-dense':
							if m <= 30:
								G = nx.gnm_random_graph(n, ERD_edges[numAdversaries], seed=None, directed=False)
							else:
								G = nx.dense_gnm_random_graph(n, ERD_edges[numAdversaries], seed=None)
							maxDegree = max([item for item in G.degree().values()])
							if nx.is_connected(G) and maxDegree <= d:
								break
						elif net == 'Erdos-Renyi-sparse':
							if m <= 30:
								G = nx.gnm_random_graph(n, ERS_edges[numAdversaries], seed=None, directed=False)
							else:
								G = nx.dense_gnm_random_graph(n, ERS_edges[numAdversaries], seed=None)
							maxDegree = max([item for item in G.degree().values()])
							if nx.is_connected(G) and maxDegree <= d:
								break
						else:
							G = nx.barabasi_albert_graph(n, m, seed=None)
							maxDegree = max([item for item in G.degree().values()])
							if maxDegree <= d:
								break
					adjacencyMatrix = [[False]*n for i in range(n)]
					graph = nx.convert.to_dict_of_lists(G)
					for node, neighbors in graph.items():
						for neighbor in neighbors:
							adjacencyMatrix[node][neighbor] = True
					node_deg = [(idx, adjacencyMatrix[idx].count(True)) for idx in range(n)]
					#node_deg.sort(key=lambda x: x[1], reverse=True)
					random.shuffle(node_deg)
					#decide which ones are adversarial
					all_nodes = [item[0] for item in node_deg]
					adversarialNodes = [item[0] for item in node_deg[:numAdversaries]]
					nonAdversarialNodes = [item[0] for item in node_deg[numAdversaries:]]
					random.shuffle(nonAdversarialNodes)
					visibleColorNodes = [item for item in nonAdversarialNodes[:numVisibles]]
					#compute reach_of_visibles
					hasBeenReached = dict.fromkeys(all_nodes, False)
					reach_of_visibles = 0
					for visibleColorNode in visibleColorNodes:
						for neighbor in G.neighbors(visibleColorNode):
							if neighbor not in adversarialNodes and neighbor not in visibleColorNodes and hasBeenReached[neighbor] == False:
								reach_of_visibles += 1
								hasBeenReached[neighbor] = True
					#compute reach_of_adversaries
					hasBeenReached = dict.fromkeys(all_nodes, False)				
					reach_of_adversaries = 0
					for adversarialNode in adversarialNodes:
						for neighbor in G.neighbors(adversarialNode):
							if neighbor not in adversarialNodes and hasBeenReached[neighbor] == False:
								reach_of_adversaries += 1
								hasBeenReached[neighbor] = True
					#compute size of largest connected component of regular nodes
					for adversary in adversarialNodes:
						G.remove_node(adversary)
					size_of_largest_cc = max(nx.connected_component_subgraphs(G), key=len).number_of_nodes()
					G.clear()
					consensus_ret.append(pd.DataFrame([(net, numVisibles, numAdversaries, reach_of_visibles, reach_of_adversaries, size_of_largest_cc)]))

	consensus_ret = pd.concat(consensus_ret)
	consensus_ret.columns = ['network', '#visible', '#adversarial', 'reach_of_visibles', 'reach_of_adversaries', 'size_of_largest_cc']

	folder = 'result/preSimStats'
	if not os.path.exists(folder):
		os.makedirs(folder)
	p = os.path.join(folder, 'preSimStats.csv')
	consensus_ret.to_csv(p, index=None)