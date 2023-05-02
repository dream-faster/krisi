from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from typing_extensions import Literal


def create_save_graphs(
    df_rolled_corr: List[pd.DataFrame],
    save_or_display: List[Literal["save", "display"]] = ["save", "display"],
    corr_direction: Literal["positive", "negative"] = "positive",
    min_correlation: float = 0.01,
    save_location: str = "output/analyse/correlations",
) -> None:
    for i, corr_df in enumerate(df_rolled_corr):
        __display_corr_graph(
            corr_df,
            save_or_display,
            corr_direction,
            min_correlation,
            save_location,
            file_name=str(i),
        )


def __display_corr_graph(
    corr: pd.DataFrame,
    save_or_display: List[Literal["save", "display"]],
    corr_direction: Literal["positive", "negative"],
    min_correlation: float,
    save_location: str,
    file_name: str = "0",
) -> None:
    feature_node_names = corr.index.values
    matrix_array = np.array(corr)
    matrix_corr = np.matrix(matrix_array, copy=False, dtype=None)

    G = nx.Graph(matrix_corr)
    G = nx.relabel_nodes(G, lambda i: feature_node_names[i])
    G.remove_edges_from(nx.selfloop_edges(G))

    __create_corr_network(
        G,
        corr_direction=corr_direction,
        min_correlation=min_correlation,
        save_location=save_location,
        file_name=file_name,
        save_or_display=save_or_display,
    )


def __create_corr_network(
    G: nx.Graph,
    corr_direction: Literal["positive", "negative"],
    min_correlation: float,
    save_location: str,
    file_name: str,
    save_or_display: List[Literal["save", "display"]],
):
    # Creates a copy of the graph
    H = G.copy()

    # Checks all the edges and removes some based on corr_direction
    for stock1, stock2, weight in G.edges(data=True):
        # if we only want to see the positive correlations we then delete the edges with weight smaller than 0
        if corr_direction == "positive":
            # it adds a minimum value for correlation.
            # If correlation weaker than the min, then it deletes the edge
            if weight["weight"] < 0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        # this part runs if the corr_direction is negative and removes edges with weights equal or largen than 0
        else:
            # it adds a minimum value for correlation.
            # If correlation weaker than the min, then it deletes the edge
            if weight["weight"] >= 0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)

    # crates a list for edges and for the weights
    edges, weights = zip(*nx.get_edge_attributes(H, "weight").items())

    # increases the value of weights, so that they are more visible in the graph
    weights = tuple([(1 + abs(x)) ** 2 for x in weights])

    # calculates the degree of each node
    # d = nx.degree(H)
    # creates list of nodes and a list their degrees that will be used later for their sizes
    # nodelist, node_sizes = zip(*d.items())

    # positions
    positions = nx.circular_layout(H)

    # Figure size
    plt.figure(figsize=(15, 15))

    # draws nodes
    nx.draw_networkx_nodes(
        H,
        positions,
        node_color="#DA70D6",
        # nodelist=nodelist,
        # the node size will be now based on its degree
        node_size=5000,
        # node_size=tuple([x**3 for x in node_sizes]),
        alpha=0.8,
    )

    # Styling for labels
    nx.draw_networkx_labels(H, positions, font_size=21, font_family="sans-serif")

    # edge colors based on weight direction
    if corr_direction == "positive":
        edge_colour = plt.cm.GnBu
    else:
        edge_colour = plt.cm.PuRd

    # draws the edges
    nx.draw_networkx_edges(
        H,
        positions,
        edgelist=edges,
        style="solid",
        # adds width=weights and edge_color = weights
        # so that edges are based on the weight parameter
        # edge_cmap is for the color scale based on the weight
        # edge_vmin and edge_vmax assign the min and max weights for the width
        width=weights,
        edge_color=weights,
        edge_cmap=edge_colour,
        edge_vmin=min(weights),
        edge_vmax=max(weights),
    )

    plt.axis("off")

    if "save" in save_or_display:
        plt.savefig(f"{save_location}/{file_name}_{corr_direction}.png", format="PNG")
    if "display" in save_or_display:
        plt.show()


def create_animation(
    path: str = "structured_data/plots", file_suffix: str = "positive"
):
    import os

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    files = os.listdir(path)

    nframes = len(files)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

    def animate(i: int):
        im = plt.imread(f"{path}/{str(i)}_{file_suffix}.png")
        plt.imshow(im)

    anim = FuncAnimation(
        plt.gcf(), animate, frames=nframes, interval=(2000.0 / nframes)
    )
    anim.save(f"{path}/output.gif", writer="imagemagick")
