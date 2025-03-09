from uxsim import *
from uxsim.OSMImporter import OSMImporter
import gymnasium as gym
import numpy as np
import random
import pickle
import torch.nn as nn
from pathlib import Path
import argparse




class Lexington:
    def __init__ (self, coordinates, traffic_flow, show):
        self.traffic_flow = traffic_flow
        self.name = f"downtown_{self.traffic_flow}"
        self.show = show
        self.action_space = None
        self.observation_space = None
        self.coordinates = coordinates

    def create(self):

        self.W = World(
                    name = f"",
                    tmax = 9600,
                    deltan = 5,
                    show_mode = 0,
                    reduce_memory_delete_vehicle_route_pref = True,
                    print_mode=0,
                    random_seed=42
                )

        nodes, links = OSMImporter.import_osm_data(
            north=self.coordinates[0], south=self.coordinates[1], east=self.coordinates[2], west=self.coordinates[3],
            custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary|secondary_link|tertiary_link"]'
        )



        # Postprocess network data
        nodes, links = OSMImporter.osm_network_postprocessing(
            nodes, links,
            node_merge_threshold=0.0005,
            node_merge_iteration=1,
            enforce_bidirectional=True 
        )

        if self.show:
            OSMImporter.osm_network_visualize(nodes, links)


        OSMImporter.osm_network_to_World(
            self.W, nodes, links,
            default_jam_density=0.2,
            coef_degree_to_meter=111000
        )



        for n in self.W.NODES_NAME_DICT:
            self.W.NODES_NAME_DICT[n].signal = [60,60]

            link_names = []

            for l in self.W.NODES_NAME_DICT[n].inlinks:
                link_name = l.split("-")[0]
                link_names.append(link_name)

            link_names = list(set(link_names))

            for l in self.W.NODES_NAME_DICT[n].inlinks:
                link_name = l.split("-")[0]

                if link_name == link_names[0]:
                    self.W.LINKS_NAME_DICT[l].signal_group = [1]
                else:
                    self.W.LINKS_NAME_DICT[l].signal_group = [0]


        dt = 30


        origins = self.W.NODES_NAME_DICT
        destinations = self.W.NODES_NAME_DICT

        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(self.W.NODES_NAME_DICT),),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=99999999999,
            shape=(len(self.W.LINKS_NAME_DICT),),
            dtype=np.float32
        )


        if self.traffic_flow == "low":
            
            for t in range(0, self.W.TMAX, dt):
                demand = random.uniform(0.25, 0.45)   
                self.W.adddemand_nodes2nodes2(origins, destinations, t, t+dt, demand)
                demand = random.uniform(0.25, 0.45)
                self.W.adddemand_nodes2nodes2(destinations, origins, t, t+dt, demand)
            

            print("-> Demand")



        if self.traffic_flow == "medium":

            for t in range(0, self.W.TMAX, dt):
                demand = random.uniform(0.45, 0.65)   
                self.W.adddemand_nodes2nodes2(origins, destinations, t, t+dt, demand)
                demand = random.uniform(0.45, 0.65)
                self.W.adddemand_nodes2nodes2(destinations, origins, t, t+dt, demand)


            print("-> Demand")




        if self.traffic_flow == "high":
                
            for t in range(0, self.W.TMAX, dt):
                demand = random.uniform(0.65, 0.85)   
                self.W.adddemand_nodes2nodes2(origins, destinations, t, t+dt, demand)
                demand = random.uniform(0.65, 0.85)
                self.W.adddemand_nodes2nodes2(destinations, origins, t, t+dt, demand)

            print("-> Demand")





        intersections = []
        links = []

        for node in self.W.NODES_NAME_DICT:
            intersections.append(self.W.NODES_NAME_DICT[node])

            for l in self.W.NODES_NAME_DICT[node].inlinks:    
                links.append(l)

        return self.W, links, intersections, self.action_space, self.observation_space
    



