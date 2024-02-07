import networkx as nx

# Step 2: Create a Graph
G = nx.Graph()

# Step 3: Add Nodes and Edges
G.add_node("n1", pos=(282,698))
G.add_node("n2", pos=(537,698))
G.add_node("n3", pos=(282,531))
G.add_node("n4", pos=(537,526))
G.add_node("n5", pos=(804,526))
G.add_node("n6", pos=(282,345))
G.add_node("n7", pos=(537,351))
G.add_node("n8", pos=(804,365))
G.add_node("n9", pos=(282,225))
G.add_node("n10", pos=(537,234))
G.add_node("n11", pos=(786,241))

G.add_edge("n1", "n2", weight=255 ) 
G.add_edge("n1", "n3", weight=167 )
G.add_edge("n2", "n4", weight=172 )
#G.add_edge("n2", "n5", weight= ) #doubt
G.add_edge("n3", "n6", weight=186 )
G.add_edge("n3", "n4", weight=255 )
G.add_edge("n4", "n7", weight=175 )
G.add_edge("n4", "n5", weight=267 )
G.add_edge("n5", "n8", weight=161 )
G.add_edge("n6", "n7", weight=255 )
G.add_edge("n8", "n11", weight=125 )
#G.add_edge("n7", "n6", weight= )
G.add_edge("n7", "n10", weight=117 )
G.add_edge("n6", "n9", weight=120 )
G.add_edge("n9", "n10", weight=255 )
G.add_edge("n10", "n11", weight=249 )

# Step 4: Find Shortest Path
shortest_path = nx.shortest_path(G, source="n1", target="n11", weight="weight")

# Step 5: Print or use the Shortest Path
print("Shortest Path:", shortest_path)