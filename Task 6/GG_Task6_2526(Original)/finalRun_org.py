"""
* Team Id: GG_2526
* Author List: Harshal Kale, Akash Mohapatra, Sharanya Anil
* Filename: finalRun_org.py
* Theme: GeoGuide
* Functions: main, signal_handler, read_priority_labels, find_shortest_path, get_next_action
* Global Variables: ip
"""

import networkx as nx
import socket
import signal
import sys
from time import sleep
import time

ip = "192.168.137.133"

"""
* Function Name: signal_handler
* Input: sig, frame
* Output: None
* Logic: Handles the SIGINT signal to perform clean-up when the program is interrupted.
* Example Call: signal.signal(signal.SIGINT, signal_handler)
"""
def signal_handler(sig, frame):
    print('Clean-up!')
    sys.exit(0)


"""
* Function Name: read_priority_labels
* Input: filename (str)
* Output: priority_labels (dict)
* Logic: Reads priority labels from a file and maps them using a fixed mapping.
* Example Call: read_priority_labels("priority_labels.txt")
"""
def read_priority_labels(filename):
    priority_labels = {}
    fixed_mapping = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E'}
    
    with open(filename, 'r') as file:
        labels = file.readline().strip().split(',')
        for label in labels:
            if label in fixed_mapping:
                #Mapping priority labels
                priority_labels[label] = fixed_mapping[label]
    return priority_labels


"""
* Function Name: find_shortest_path
* Input: graph (networkx.Graph), source (str), target (str)
* Output: shortest_path (list)
* Logic: Finds the shortest path in the graph from source to target using Dijkstra's algorithm.
* Example Call: find_shortest_path(G, "A", "B")
"""
def find_shortest_path(graph, source, target):
    return nx.shortest_path(graph, source=source, target=target, weight="weight")


"""
* Function Name: get_next_action
* Input: G (networkx.Graph), curr_pos (tuple), next_pos (tuple), curr_dir (str)
* Output: direction, action (tuple)
* Logic: Determines the next direction and action based on current and next positions and current direction.
* Example Call: get_next_action(G, "n1", "E", "N")
"""
def get_next_action(G,curr_pos, next_pos, curr_dir):
    dx = next_pos[0] - curr_pos[0]
    dy = next_pos[1] - curr_pos[1]

    if curr_pos == G.nodes["n11"]['pos'] and next_pos == G.nodes["E"]['pos']:
        return 'S', '1'
    
    #1-Forward, 3-Right, 2-Left, 4-Turn around and Forward,9-Information sent by Arudino by Python that Node has reached 
    #N=North,S-South,E-East,W-West
    if curr_dir == 'N':  # Initially facing North
        if dx > 0: return 'E', '3'
        elif dx < 0: return 'W', '2'
        elif dy > 0: return 'N', '1'
        elif dy < 0: return 'S', '4' 
    elif curr_dir == 'E':  # Initially facing East
        if dx > 0: return 'E', '1'
        elif dx < 0: return 'W', '4'
        elif dy > 0: return 'N', '2'
        elif dy < 0: return 'S', '3'
    elif curr_dir == 'W':  # Initially facing West
        if dx > 0: return 'E', '4'
        elif dx < 0: return 'W', '1'
        elif dy > 0: return 'N', '3'
        elif dy < 0: return 'S', '2'
    elif curr_dir == 'S':  # Initially facing South
        if dx > 0: return 'E', '2'
        elif dx < 0: return 'W', '3'
        elif dy > 0: return 'N', '4'
        elif dy < 0: return 'S', '1'


"""
* Function Name: main
* Input: None
* Output: None
* Logic: The main function to control the robot's movement based on priority labels and predefined paths.
* Example Call: main()
"""
def main():
    # wait_times: Dictionary defining waiting times for each priority label
    wait_times = {'A': 3, 'B': 4.5, 'C': 2.5, 'D': 1, 'E': 13.5}
    
    priority_labels = read_priority_labels("priority_labels.txt")
    
    # Making graph using the library for finding out the shortest path from one point to another
    G = nx.Graph()

    G.add_node("n1", pos=(282, 225))
    G.add_node("n2", pos=(537, 225))
    G.add_node("n3", pos=(282, 345))
    G.add_node("n4", pos=(537, 345))
    G.add_node("n5", pos=(804, 345))
    G.add_node("n6", pos=(282, 531))
    G.add_node("n7", pos=(537, 531))
    G.add_node("n8", pos=(804, 531))
    G.add_node("n9", pos=(282, 698))
    G.add_node("n10", pos=(537, 698))
    G.add_node("n11", pos=(804, 698))
    G.add_node("E", pos=(282, 730))
    G.add_node("B", pos=(670, 345))
    G.add_node("A", pos=(370, 225))
    G.add_node("D", pos=(370, 531))
    G.add_node("C", pos=(670, 531))
    
    G.add_edge("n1", "A", weight = 100)
    G.add_edge("n1", "n3", weight=172 )
    G.add_edge("n2", "n4", weight=172 )
    G.add_edge("n2", "A", weight = 155)
    G.add_edge("n3", "n6", weight=175 )
    G.add_edge("n3", "n4", weight=255 )
    G.add_edge("n4", "n7", weight=175 )
    G.add_edge("n4", "n5", weight=267 )
    G.add_edge("n4", "B", weight =133.5)
    G.add_edge("n5", "n8", weight = 175 )
    G.add_edge("n5", "B", weight = 133.5)
    G.add_edge("n6", "D", weight = 100)
    G.add_edge("n8", "n11", weight=120 )
    G.add_edge("n8", "C", weight = 133.5)
    G.add_edge("n7", "n10", weight=120 )
    G.add_edge("n7", "D", weight = 155)
    G.add_edge("n7", "C", weight = 133.5)
    G.add_edge("n6", "n9", weight=120 )
    G.add_edge("n9", "n10", weight=255 )
    G.add_edge("n9", "E", weight = 120)
    G.add_edge("n10", "n11", weight=267)
    G.add_edge("E", "n11", weights = 240)
    
    current_node = "n1"
    directions=[]
    
    # Connection and communication from python to arduino and vice versa
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.connect((ip, 8002))
        
        print("Connected to ESP32")
        
        while True:
            signal.signal(signal.SIGINT, signal_handler)
            try:
                while True:
                    curr_dir = 'N'  # Initially facing North
                    for j,label in enumerate(priority_labels):
                        next_node = priority_labels[label]
                        shortest_path = find_shortest_path(G, current_node, next_node)
                       
                        if(current_node=="A" or current_node=="C" or current_node=="E"):
                            shortest_path.pop(0)
                             
                        for i in range(len(shortest_path) - 1):
                            curr_pos = G.nodes[shortest_path[i]]['pos']
                            next_pos = G.nodes[shortest_path[i + 1]]['pos']
                            curr_dir, action = get_next_action(G,curr_pos, next_pos, curr_dir)
                            directions.append(action)
    
                        directions.append('0')
                        
                        for i,direction in enumerate(directions):
                            if i == len(directions) - 1:
                                sleep(wait_times[label])
                                s.sendall(direction.encode())
                            else:
                                while True: 
                                    data = s.recv(1024)
                                    received_data = data.decode().strip()
                                    if received_data == "9":
                                        s.sendall(direction.encode())
                                        break
                                
                        directions=[]
                        current_node = next_node


                    if(j==len(priority_labels)-1):
                        current_node = list(priority_labels.keys())[-1]
                        next_node= "n1"
                        shortest_path = find_shortest_path(G, current_node, next_node)
                        
                        for i in range(len(shortest_path) - 1):
                            curr_pos = G.nodes[shortest_path[i]]['pos']
                            next_pos = G.nodes[shortest_path[i + 1]]['pos']
                            curr_dir, action = get_next_action(G, curr_pos, next_pos, curr_dir)
                            directions.append(action)
                            
                        directions.append('1')
                        
                        for i,direction in enumerate(directions):
                            while True: 
                                data = s.recv(1024)
                                received_data = data.decode().strip()
                                
                                if received_data == "9": 
                                    s.sendall(direction.encode())
                                    break
                        sleep(1.5)
                        s.sendall(b'5')          
                        directions=[]
                    break
                break                     
            except KeyboardInterrupt:
                pass
        
if __name__ == "__main__":
    main()