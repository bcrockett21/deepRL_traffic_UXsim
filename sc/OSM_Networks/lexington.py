from uxsim import *
from uxsim.OSMImporter import OSMImporter
import gymnasium as gym
import numpy as np
import random
import pickle
import torch.nn as nn
from pathlib import Path
import argparse
from itertools import product
import warnings

warnings.filterwarnings("ignore", category=UserWarning)




class Lexington:
    def __init__ (self, coordinates, traffic_flow, netowrk_size, show):
        self.traffic_flow = traffic_flow
        self.network_size = netowrk_size
        self.name = f"{netowrk_size}_{self.traffic_flow}"
        self.animation_name = "Users/blakecrockett/Documents/ds_capstone/charts/" + self.name + "_"
        self.show = show
        self.action_space = None
        self.observation_space = None
        self.coordinates = coordinates
        self.nodes, self.links = self.initialize()


    def initialize(self):
        nodes, links = OSMImporter.import_osm_data(
            north=self.coordinates[0], south=self.coordinates[1], east=self.coordinates[2], west=self.coordinates[3],
            custom_filter='["highway"~"motorway|trunk|primary|secondary|tertiary"]'
        )

        if self.network_size == "downtown":
            nodes, links = OSMImporter.osm_network_postprocessing(
                nodes, links,
                node_merge_threshold=0.001,
                node_merge_iteration=3,
                enforce_bidirectional=True 
            )

        else:

            nodes, links = OSMImporter.osm_network_postprocessing(
                nodes, links,
                node_merge_threshold=0.005,
                node_merge_iteration=5,
                enforce_bidirectional=True
            )

        delete = []
        for n in nodes:
            inlinks = []
            for l in links:
                if l[2] == n[0]:
                    inlinks.append([l[1], l[2]])
                    if len(inlinks) > 4:
                        delete.append([l[1], l[2]])
                        delete.append([l[2], l[1]])

        for l in links:
            for d in delete:
                if (l[1] == d[0]) and (l[2] == d[1]):
                    if l in links:
                        links.remove(l)

        if self.show:
            OSMImporter.osm_network_visualize(nodes, links)
        
        return nodes, links


    def generate_animation(self):
        self.W.exec_simulation()
        self.W.analyzer.print_simple_stats()
        

        #print("-> Simple Animation")
        #self.W.analyzer.network_anim(animation_speed_inverse=5, timestep_skip=30, detailed=0, network_font_size=0, figsize=(4,4), file_name="/Users/blakecrockett/Documents/ds_capstone/charts/simple.gif")
        
        print("-> Fancy Animation")
        self.W.analyzer.network_fancy(animation_speed_inverse=1, 
                                    sample_ratio=1, 
                                    interval=5, 
                                    trace_length=5,
                                    file_name = f"/Users/blakecrockett/Documents/ds_capstone/charts/{self.name}_fancy.gif")



    def create(self, animate):
        self.animate = animate

        self.W = World(
                    name = f"",
                    tmax = 256*30,
                    deltan = 5,
                    show_mode = 1,
                    reduce_memory_delete_vehicle_route_pref = True,
                    print_mode=0,
                    random_seed=42
                )

        OSMImporter.osm_network_to_World(
            self.W, self.nodes, self.links,
            default_jam_density=0.2,
            coef_degree_to_meter=111000
        )
        


        for n in self.W.NODES_NAME_DICT:
                
                self.W.NODES_NAME_DICT[n].signal = [60,60]
                link_names = []

                i = 0
                for l in self.W.NODES_NAME_DICT[n].inlinks:
                    if len(self.W.NODES_NAME_DICT[n].inlinks) == 2:
                        self.W.LINKS_NAME_DICT[l].signal_group = [0]
                    else:
                        if i % 2 == 0:
                            self.W.LINKS_NAME_DICT[l].signal_group =[1]
                        else:
                            self.W.LINKS_NAME_DICT[l].signal_group =[0]
                        self.W.LINKS_NAME_DICT[l].name = str(self.W.LINKS_NAME_DICT[l].signal_group)
                        i += 1

              
        
      """
        self.W.show_network(
                width=1,
                left_handed=0,
                figsize=(16, 16),
                network_font_size=8,
                node_size=8,
                show_id=True
            )
        """
                    




        nodes = self.W.NODES_NAME_DICT
        destinations = self.W.NODES_NAME_DICT



        observation_space = gym.spaces.Box(
            low=0,
            high=99999999999,
            shape=(len(links),),
            dtype=np.float32
        )

        action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(intersections),),
            dtype=np.float32
        )

        self.observation_space = observation_space
        self.action_space = action_space



        dt = 30
        y2 = 38.047
        x2 = -84.497

        # DOWNTOWN ONLY

        if self.network_size == "downtown":
            x_coordinates = [-84.49, -84.504]
            y_coordinates = [38.04, 38.054]

            if self.traffic_flow == "low":
                """
                for t in range(0, self.W.TMAX, dt):
                    demand = random.uniform(0.25, 0.45)   
                    self.W.adddemand_nodes2nodes2(nodes, destinations, t, t+dt, demand)
                    demand = random.uniform(0.25, 0.45)
                    self.W.adddemand_nodes2nodes2(destinations, nodes, t, t+dt, demand)
                """
                lower = 0.25
                upper = 0.45
                
                for t in range(0, int(self.W.TMAX / 2), dt):
                    # high demand entering downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.05, t, t+dt, flow=demand)
                    # low demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 3
                        self.W.adddemand_area2area2(x2, y2, 0.05, x1, y1, 0.1, t, t+dt, flow=demand)


                for t in range(int(self.W.TMAX / 2), self.W.TMAX, dt):
                    # low demand entering downtown  
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 3
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.05, t, t+dt, flow=demand)
                    # high demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x2, y2, 0.05, x1, y1, 0.1, t, t+dt, flow=demand)
                


            if self.traffic_flow == "medium":
                lower = 0.45
                upper = 0.65
                
                for t in range(0, int(self.W.TMAX / 2), dt):
                    # high demand entering downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.025, t, t+dt, flow=demand)
                    # low demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 3
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.1, t, t+dt, flow=demand)


                for t in range(int(self.W.TMAX / 2), self.W.TMAX, dt):
                    # low demand entering downtown  
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 3
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.025, t, t+dt, flow=demand)
                    # high demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.1, t, t+dt, flow=demand)



            if self.traffic_flow == "high":
                lower = 0.65
                upper = 0.85
                
                for t in range(0, int(self.W.TMAX / 2), dt):
                    # high demand entering downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.025, t, t+dt, flow=demand)
                    # low demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 3
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.1, t, t+dt, flow=demand)


                for t in range(int(self.W.TMAX / 2), self.W.TMAX, dt):
                    # low demand entering downtown  
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 3
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.025, t, t+dt, flow=demand)
                    # high demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.1, t, t+dt, flow=demand)



        # FULL LEXINGTON GRID


        else:
            #print("LEXINGTON")
            x_coordinates = [-84.425, -84.60]
            y_coordinates = [38.08, 37.98]

            if self.traffic_flow == "low":
                lower = 0.25
                upper = 0.45
                
                for t in range(0, int(self.W.TMAX / 2), dt):
                    # high demand entering downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x1, y1, 0.1, x2, y2, 0.03, t, t+dt, flow=demand)
                    # low demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 4
                        self.W.adddemand_area2area2(x2, y2, 0.03, x1, y1, 0.1, t, t+dt, flow=demand)


                for t in range(int(self.W.TMAX / 2), self.W.TMAX, dt):
                    # low demand entering downtown  
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 4
                        self.W.adddemand_area2area2(x1, y1, 0.075, x2, y2, 0.025, t, t+dt, flow=demand)
                    # high demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.075, t, t+dt, flow=demand)


            if self.traffic_flow == "medium":
                lower = 0.45
                upper = 0.65
                
                for t in range(0, int(self.W.TMAX / 2), dt):
                    # high demand entering downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x1, y1, 0.075, x2, y2, 0.025, t, t+dt, flow=demand)
                    # low demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 4
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.075, t, t+dt, flow=demand)


                for t in range(int(self.W.TMAX / 2), self.W.TMAX, dt):
                    # low demand entering downtown  
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 4
                        self.W.adddemand_area2area2(x1, y1, 0.075, x2, y2, 0.025, t, t+dt, flow=demand)
                    # high demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.075, t, t+dt, flow=demand)


            if self.traffic_flow == "high":
                lower = 0.65
                upper = 0.85
                
                for t in range(0, int(self.W.TMAX / 2), dt):
                    # high demand entering downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x1, y1, 0.075, x2, y2, 0.025, t, t+dt, flow=demand)
                    # low demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 4
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.075, t, t+dt, flow=demand)


                for t in range(int(self.W.TMAX / 2), self.W.TMAX, dt):
                    # low demand entering downtown  
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper) / 4
                        self.W.adddemand_area2area2(x1, y1, 0.075, x2, y2, 0.025, t, t+dt, flow=demand)
                    # high demand leaving downtown
                    for x1, y1 in product(x_coordinates, y_coordinates):
                        demand = random.uniform(lower, upper)
                        self.W.adddemand_area2area2(x2, y2, 0.025, x1, y1, 0.075, t, t+dt, flow=demand)




        #print("-> Demand")

        intersections = []
        links = []

        for node in self.W.NODES_NAME_DICT:
            intersections.append(self.W.NODES_NAME_DICT[node])

        for l in self.W.LINKS_NAME_DICT:    
            links.append(self.W.LINKS_NAME_DICT[l])

        if self.animate:
            self.generate_animation()

        return self.W, links, intersections, self.action_space, self.observation_space
    






