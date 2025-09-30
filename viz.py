import random

import networkx as nx
import matplotlib.pyplot as plt

def drawGenomeGraph(genome):
    G = nx.Graph()
    nodes = []
    connectionsOut = {}
    connectionsIn = {}
    for gene in genome['genes']:

        if gene['enabled']:
            G.add_edge(gene['into'], gene['out'])
            if gene['into'] not in nodes:
                nodes.append(gene['into'])
                connectionsOut[gene['into']] = [gene['out']]

                if gene['out'] not in connectionsIn:
                    connectionsIn[gene['out']] = [gene['into']]
                else:
                    connectionsIn[gene['out']].append(gene['into'])


            if gene['out'] not in nodes:
                nodes.append(gene['out'])

                if gene['out'] not in connectionsIn:
                    connectionsIn[gene['out']] = [gene['into']]


                if gene['out'] not in connectionsOut:
                    connectionsOut[gene['into']] = [gene['out']]
                else:
                    connectionsOut[gene['into']].append(gene['out'])

        else:
            if gene['out'] not in list(G.nodes) and gene['out'] not in nodes:
                G.add_node(gene['out'])
                nodes.append(gene['out'])

            if gene['into'] not in list(G.nodes) and gene['into'] not in nodes:
                G.add_node(gene['into'])
                nodes.append(gene['into'])

        # G.add_node("PENIS")

    nodes.sort()

    # explicitly set positions
    pos = {}
    # pos["PENIS"] = (100, 100)
    for i in range(17):
        pos[i] = (0, i*150)

    for o in range(0, 4):
        pos[1000000 + o] = (1000, (o * 300) + (16*150)/4)

    for n in nodes:
        if n not in pos:
            pos[n] = (random.randint(1, 1000) + random.random(),
                      150*random.randint(0, 16) + random.random())

            # x = 0
            # y = 0

            # if n in connectionsOut and n in connectionsIn:
            #     for out in connectionsOut[n]:
            #         x += pos[out][0]
            #         y += pos[out][1]
            #
            #     for ins in connectionsIn[n]:
            #         x += pos[ins][0]
            #         y += pos[ins][1]
            #
            #     pos[n] = (x/(len(connectionsOut[n])+ len(connectionsIn[n])),
            #               y/(len(connectionsOut[n])+ len(connectionsIn[n])))
            #
            # elif n in connectionsOut:
            #     for out in connectionsOut[n]:
            #         x += pos[out][0]
            #         y += pos[out][1]
            #
            #     pos[n] = (x / (len(connectionsOut[n])),
            #               y / (len(connectionsOut[n])))
            #
            # elif n in connectionsIn:
            #
            #     for ins in connectionsIn[n]:
            #         x += pos[ins][0]
            #         y += pos[ins][1]
            #
            #     pos[n] = (x / (len(connectionsIn[n])),
            #               y / ( len(connectionsIn[n])))




    options = {
        "font_size": 10,
        "node_size": 300,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 5,
    }

    nx.draw_networkx(G, pos, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    return plt
