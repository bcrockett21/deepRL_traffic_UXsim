import random
from uxsim import *
import pickle
from pathlib import Path
import gymnasium as gym
import numpy as np


random.seed = 42





class MediumNetwork:
    def __init__(self, traffic_flow, model):
        self.traffic_flow = traffic_flow
        self.model = model
        self.name = f"{self.model}_{self.traffic_flow}_medium"

        self.action_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(8,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=9999999,
            shape=(50,),
            dtype=np.float32
        )


    def createAndSave(self):

            self.W = World(
                        name = f"{self.name}",
                        tmax = 256*30,
                        deltan = 5,
                        show_mode = 1,
                        reduce_memory_delete_vehicle_route_pref = True,
                        print_mode=0,
                        random_seed=42
                    )

            # start node
            self.s = self.W.addNode("s", 0, 0)

            # intersections
            self.I1 = self.W.addNode("I1", 0, 1, signal=[60,60])
            self.I2 = self.W.addNode("I2", 0, 2, signal=[60,60])
            self.I3 = self.W.addNode("I3", 0, 3, signal=[60,60])
            self.I4 = self.W.addNode("I4", 0, 4, signal=[60,60])
            self.I5 = self.W.addNode("I5", 0, 5, signal=[60,60])
            self.I6 = self.W.addNode("I6", 0, 6, signal=[60,60])
            self.I7 = self.W.addNode("I7", 0, 7, signal=[60,60])
            self.I8 = self.W.addNode("I8", 0, 8, signal=[60,60])


            self.intersections = [self.I1, self.I2, self.I3, self.I4,
                                self.I5, self.I6, self.I7, self.I8]
            

            # end node
            self.t = self.W.addNode("t", 0, 9)

            # side streets
            self.E1 = self.W.addNode("E1", 1, 1)
            self.W1 = self.W.addNode("W1", -1, 1)
            self.E2 = self.W.addNode("E2", 1, 2)
            self.W2 = self.W.addNode("W2", -1, 2)
            self.E3 = self.W.addNode("E3", 1, 3)
            self.W3 = self.W.addNode("W3", -1, 3)
            self.E4 = self.W.addNode("E4", 1, 4)
            self.W4 = self.W.addNode("W4", -1, 4)
            self.E5 = self.W.addNode("E5", 1, 5)
            self.W5 = self.W.addNode("W5", -1, 5)
            self.E6 = self.W.addNode("E6", 1, 6)
            self.W6 = self.W.addNode("W6", -1, 6)
            self.E7 = self.W.addNode("E7", 1, 7)
            self.W7 = self.W.addNode("W7", -1, 7)
            self.E8 = self.W.addNode("E8", 1, 8)
            self.W8 = self.W.addNode("W8", -1, 8)
            




            self.cross_streets = [[self.E1, self.I1], [self.W1, self.I1], [self.E2, self.I2], [self.W2, self.I2], 
                        [self.E3, self.I3], [self.W3, self.I3], [self.E4, self.I4], [self.W4, self.I4], [self.E5, self.I5],  
                        [self.W5, self.I5], [self.E6, self.I6], [self.W6, self.I6], [self.E7, self.I7], [self.W7, self.I7],  
                        [self.E8, self.I8], [self.W8, self.I8]]
                




            #E <-> W direction: signal group 0
            for n1,n2 in self.cross_streets:
                
                self.W.addLink(f"{n1.name}_{n2.name}", n1, n2, length=200, free_flow_speed=15.6464, signal_group=0)
                self.W.addLink(f"{n2.name}_{n1.name}", n2, n1, length=200, free_flow_speed=15.6464, signal_group=0)


            # N <-> S direction: signal group 

            # I6 <-> I1
            self.W.addLink(f"{self.s}_{self.I1}", self.s, self.I1, length=236.17, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I1}_{self.s}", self.I1, self.s, length=236.17, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I1 <-> I2
            self.W.addLink(f"{self.I1}_{self.I2}", self.I1, self.I2, length=331.58, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I2}_{self.I1}", self.I2, self.I1, length=331.58, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I2 <-> I3
            self.W.addLink(f"{self.I2}_{self.I3}", self.I2, self.I3, length=171.07, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I3}_{self.I2}", self.I3, self.I2, length=171.07, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I3 <-> I4
            self.W.addLink(f"{self.I3}_{self.I4}", self.I3, self.I4, length=282.83, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I4}_{self.I3}", self.I4, self.I3, length=282.83, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I4 <-> I5
            self.W.addLink(f"{self.I4}_{self.I5}", self.I4, self.I5, length=303.66, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I5}_{self.I4}", self.I5, self.I4, length=303.66, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I5 <-> I6
            self.W.addLink(f"{self.I5}_{self.I6}", self.I5, self.I6, length=235.54, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I6}_{self.I5}", self.I6, self.I5, length=235.54, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I6 <-> I7
            self.W.addLink(f"{self.I6}_{self.I7}", self.I6, self.I7, length=421.81, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I7}_{self.I6}", self.I7, self.I6, length=421.81, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I7 <-> I8
            self.W.addLink(f"{self.I7}_{self.I8}", self.I7, self.I8, length=266.13, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.I4}_{self.I7}", self.I8, self.I7, length=266.13, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

            # I8 <-> t
            self.W.addLink(f"{self.I8}_{self.t}", self.I8, self.t, length=320.97, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)
            self.W.addLink(f"{self.t}_{self.I8}", self.t, self.I8, length=320.97, free_flow_speed=20.1164, number_of_lanes=3, signal_group=1)

           
            self.links = self.W.LINKS_NAME_DICT




            self.main_road_links = [[self.s, self.I1], [self.I1, self.I2], [self.I2, self.I3], [self.I3, self.I4],
                            [self.I4, self.I5], [self.I5, self.I6], [self.I6, self.I7], [self.I7, self.I8], [self.I8, self.t]]
            

            self.nodes = self.W.NODES_NAME_DICT

            origins = []
            destinations = []

            for n in self.nodes:
                if n.startswith("I"):
                    pass
                else:
                    origins.append(n)
                    destinations.append(n)






            # add demands based on selected traffic level

            dt = 30

            if self.traffic_flow == "low":

                fname = f"/Users/blakecrockett/Documents/ds_capstone/scenarios/{self.traffic_flow}_medium"


                for t in range(0, self.W.TMAX, dt):
                    demand = random.uniform(0.25, 0.45)   
                    self.W.adddemand_nodes2nodes2(origins, destinations, t, t+dt, demand)
                    demand = random.uniform(0.25, 0.45)
                    self.W.adddemand_nodes2nodes2(destinations, origins, t, t+dt, demand)
                


                #self.W.save(fname)




            if self.traffic_flow == "medium":

                fname = f"/Users/blakecrockett/Documents/ds_capstone/scenarios/{self.traffic_flow}_medium"

                for t in range(0, self.W.TMAX, dt):
                    demand = random.uniform(0.45, 0.65)   
                    self.W.adddemand_nodes2nodes2(origins, destinations, t, t+dt, demand)
                    demand = random.uniform(0.45, 0.65)
                    self.W.adddemand_nodes2nodes2(destinations, origins, t, t+dt, demand)



                #self.W.save(fname)





            if self.traffic_flow == "high":

                fname = f"/Users/blakecrockett/Documents/ds_capstone/scenarios/{self.traffic_flow}_medium"

                for t in range(0, self.W.TMAX, dt):
                    demand = random.uniform(0.65, 0.85)   
                    self.W.adddemand_nodes2nodes2(origins, destinations, t, t+dt, demand)
                    demand = random.uniform(0.65, 0.85)
                    self.W.adddemand_nodes2nodes2(destinations, origins, t, t+dt, demand)


                #self.W.save(fname)





    def load_network(self, show):
        self.createAndSave()

        

        if show:
            self.W.show_network(show_id=False,
                           figsize=(8, 8),
                            node_size=8,
                            left_handed=0,
                           network_font_size=0)
        
        intersections = []
        links = []
        for node in self.W.NODES_NAME_DICT:
            if node.startswith("I"):
                intersections.append(self.W.NODES_NAME_DICT[node])
            for l in self.W.NODES_NAME_DICT[node].inlinks:    
                links.append(l)

        return self.W, links, intersections, self.action_space, self.observation_space
    


