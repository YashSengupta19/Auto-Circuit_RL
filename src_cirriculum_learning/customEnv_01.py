import gymnasium as gym  # The base class for any RL environment
from gymnasium import spaces  # Helps us in defining action and observation spaces
import numpy as np
import networkx as nx  # A graph library to model our circuits as graphs
import csvProcessor_01 as rd
import extract_features_03 as fe
import subprocess
import os


simulation_counter = 0
unique_counter = 0
good_circuits = []

class CircuitEnv(gym.Env):  # Custom Environment


    def __init__(self, max_components=15, value_buckets=5, folder_path = "../data"):
        super(CircuitEnv, self).__init__()

        self.max_components = max_components  # Maximum number of components possible in a circuit.
        self.value_buckets = value_buckets  # Maximum number of possible values each compomnent can have. 
        self.node_counter = 2  # 0 = GND, 1 = VDD --> Each circuit starts with 2 nodes (GND and VDD)
        self.episode_number = 0 # This value is incremented after each episode
 
        
        # We can define custom buckets later for R, L, C
        self.bucket_ranges = {
            0: np.array([1000, 2000, 3000, 4000, 5000]),     # R: 1Ω to 1MΩ
            2: np.array([0.1, 0.5, 1.0, 2.0, 2.5]),   # L: 1uH to 1mH
            1: np.array([3*1e-6, 2*1e-6, 1e-6, 4*1e-6, 5*1e-6])   # C: 1pF to 1uF
        }
        
        self.component_list = ["res", "cap", "ind"]

        # Action = [component_type, value_index, node1, node2]
        # 0 --> R, 1 --> L, 2 --> C  component_type
        # Each value_bucket contains a list of values for each component. The action space will specify the index of the value_buckets[]. Based on the index we will take the value from the bucket.
        # The next two values are the node indices (Specifies where the component can be placed in the circuit)
        self.action_space = spaces.MultiDiscrete([3, value_buckets, max_components, max_components])


        self.observation_space = spaces.Dict({
            "components": spaces.Box(-1, 1, shape=(self.max_components, 4), dtype=np.float32), # Specifies the circuit structure 
            "feature_vector": spaces.Box(-np.inf, np.inf, shape=(86,), dtype=np.float32), # Specifies the location of target feature vector for a circuit
        })

        self.processor = rd.CSVProcessor(folder_path) # This reads each file from the folder to be used as the trarget graph for that episode
        
        self.LTSpice =  r"C:\Users\jishu\AppData\Local\Programs\ADI\LTspice\LTspice.exe" # LT Spice executable file path

        self.reset() # Calls the reset() method
    
    def update_Graph(self, new_node_value, old_node_value, node2_flag): 
        
        '''
        When a new node is added to the circuit, then the entire graph G needs to be updated and
        the self.components also needs to be updated.
        '''
        new_components = [] # Start with a new components list
        node_value = None
        for item in self.components:
            if node2_flag:
                if item[3] == old_node_value:
                    node_value = new_node_value
                else:
                    node_value = item[3]
                new_components.append((item[0], item[1], item[2], node_value))
            else:
                if item[2] == old_node_value:
                    node_value = new_node_value
                else:
                    node_value = item[2]
                new_components.append((item[0], item[1], node_value, item[3]))
            
        self.components = new_components # Now the self.components is also updated. But this self.components still doesn't consists of the extra added node.
        
        # Now we have to use this updated self.component to update the Graph G
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from([0, 1])
        for u, v, data in self.G.edges(data=True):
            edge_type = data['type']
            edge_value = data['value']
            if v == old_node_value:
                v = new_node_value
                
            new_graph.add_edge(u, v, type=edge_type, value=edge_value)
                
            # Handle node assignment
            for node in [u, v]:
                if node not in new_graph.nodes:
                    new_graph.add_node(node)  # Add the node to the graph representing the circuit
        return new_graph
                
   
    

    # LT Spice Interfacing Code
    def build_circuit(self, component_list, filename="generated_circuit.asc"):
        nodes = set()
        for comp in component_list:
            n1, n2 = str(comp[2]), str(comp[3])
            if n1 != '0': 
                nodes.add(n1)
            if n2 != '0': 
                nodes.add(n2)

        sorted_nodes = sorted(nodes, key=int)
        node_index_map = {node: idx for idx, node in enumerate(sorted_nodes)}

        wires = []
        for idx in range(len(sorted_nodes)):
            x = idx * 112
            wires.append(f"WIRE {x} -168 {x} {(len(component_list)) * 80}")

        component_blocks = []
        wire_blocks = []
        flag_blocks = []
        delta = 0
        counts = {"res": 0, "cap": 0, "ind": 0, "voltage": 0}

        for comp in component_list:
            ctype, value, node1, node2 = comp
            node1, node2 = str(node1), str(node2)

            if ctype not in counts:
                counts[ctype] = 0
            counts[ctype] += 1
            inst_name = f"{ctype[0].upper()}{counts[ctype]}"

            if node1 != '0' and node2 != '0':
                idx1 = node_index_map[node1]
                idx2 = node_index_map[node2]
                smaller_idx = min(idx1, idx2)
            else:
                non_zero_node = node2 if node1 == '0' else node1
                smaller_idx = node_index_map[non_zero_node]

            x = smaller_idx * 112 + (64 if ctype == "cap" else 96)
            y = -16 + 80 * delta
            delta += 1

            component_blocks.extend([
                f"SYMBOL {ctype} {x} {y} R90",
                f"SYMATTR InstName {inst_name}",
                f"SYMATTR Value {value}"
            ])  # adding a component

            if node1 == '0' or node2 == '0':
                offset = 0 if ctype == "cap" else 16
                flag_blocks.append(f"FLAG {x - offset} {y + 16} 0")
            else:
                x1 = x - (0 if ctype == "cap" else 16)
                x2 = max(idx1, idx2) * 112
                wire_blocks.append(f"WIRE {x1} {y+16} {x2} {y+16}")

        # # --- Add ground to the last node in sorted_nodes ---
        # if len(sorted_nodes) > 2:  # only if non-empty
        #     last_node = sorted_nodes[-1]
        #     idx_last = node_index_map[last_node]
        #     x = idx_last * 112
        #     y = (len(component_list)) * 80   # place ground below last wire
        #     flag_blocks.append(f"FLAG {x} {y} 0")

        with open(filename, 'w') as f:
            # default commands for initialization and voltage source
            f.write("Version 4\nSHEET 1 880 680\n")
            f.write("SYMBOL voltage -272 -176 R0\n")
            f.write("WINDOW 123 0 0 Left 0\n")
            f.write("WINDOW 39 0 0 Left 0\n")
            f.write("SYMATTR InstName AC1\n")
            f.write("SYMATTR Value AC 1\n")
            f.write("FLAG -272 -80 0\n")
            f.write("WIRE 0 -160 -272 -160\n")
            f.write("TEXT -380 328 Left 2 !.ac oct 10 1 1000\n")
            
            if len(sorted_nodes) > 2:  # only if non-empty and there are more than 2 nodes in the circuit
                last_node = sorted_nodes[-1]
                idx_last = node_index_map[last_node]
                x = idx_last * 112
                flag_blocks.append(f"FLAG {x} {-144} Vout")  # Place the Vout always on the last node
                
            # Always add ground to the second node
            if len(sorted_nodes) > 1:  # only if non-empty
                y = (len(component_list)) * 80   # place ground below last wire
                flag_blocks.append(f"FLAG {112} {y} 0")
                
            # ac analysis: 10 points per decade, 1Hz to 1000Hz
            for wire in wires:
                f.write(wire + "\n")
            for line in component_blocks:
                f.write(line + "\n")
            for line in wire_blocks:
                f.write(line + "\n")
            for line in flag_blocks:
                f.write(line + "\n")

        # print(f".asc file generated: {filename}")


    
    def run_simulation(self, circuit_path, ltspice_path):
        
        
        global simulation_counter
        simulation_counter += 1
        
        counter_file = "simulation_counter.txt"
        with open(counter_file, "w") as f:
            f.write(str(simulation_counter))
        # Add -b (batch) and -run to ensure simulation runs in the background without waiting for user input
        command = [ltspice_path, "-b", "-run", circuit_path]
        # print(f"Running LTSpice simulation with command: {' '.join(command)}")

        try:
            # Run the LTSpice simulation and capture stdout and stderr
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)

            # Print the output from LTSpice (stdout and stderr)
            if result.stdout:
                print("LTSpice stdout:", result.stdout)
            if result.stderr:
                print("LTSpice stderr:", result.stderr)
                
            # Check the return code to see if the simulation was successful
            if result.returncode != 0:
                # print("Simulation completed successfully.")
                print(f"Simulation failed with return code: {result.returncode}")
                print("Error:", result.stderr)
                
            # else:
            #     print(f"Simulation failed with return code: {result.returncode}")
            #     print("Error:", result.stderr)

            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running LTSpice: {e}")
            print(f"Output: {e.output}")
            print(f"Error: {e.stderr}")
            return None
        
    
    
    
    def plot_response(self, raw_file_path, output_csv="frequency_response.csv"):

        from PyLTSpice import RawRead
        import numpy as np
        import csv
        import extract_features_03 as fe  # make sure this import matches your actual module

        # Load the raw file
        ltr = RawRead(raw_file_path)

        # Get trace names
        trace_names = ltr.get_trace_names()


        # Get frequency trace
        freq = ltr.get_trace("frequency")
        f_vals = np.real(freq.get_wave(0))

        # Try to get Vout trace
        if "V(vout)" in trace_names:
            vout = ltr.get_trace("V(vout)")
            vout_vals = vout.get_wave(0)

            # Compute gain and phase
            gain = np.abs(vout_vals)
            phase_rad = np.angle(vout_vals)

            # Save to CSV
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Frequency", "Gain", "Phase"])
                for f, g, p in zip(f_vals, gain, phase_rad):
                    writer.writerow([f, g, p])

            # print(f"Frequency response saved to '{output_csv}'")

            # Extract features from CSV file
            featureVector, svdFailed = fe.extractFeatures(file_path=output_csv)
        else:
            
            featureVector, svdFailed = [], True

        return featureVector, svdFailed

    
    # LT Spice Interfacing Code ends...
    
    

    def reset(self, *, seed=None, options=None): 
        
        '''This method is called at the start of every episode and sets certain initial conditions'''
        
        super().reset(seed=seed)
        
        # self.components = [[-1, -1, -1, -1]]*15 
        self.components = [] 
        '''
        List of components as tuples (type, value_idx, n1, n2)  --> This is what will be our circuit representation after every action ois taken place
        Initially the self.components contains all -1 indicating the absence of any component in the circuit.
        '''
        
        # G is a graphical representation of the same circuit
        self.G = nx.DiGraph()
        self.G.add_nodes_from([0, 1])  # 0 = GND, 1 = VDD
        self.node_counter = 2
    
        self.featureVector, self.svdFailed = self.processor.process_next() # This stores the target pole/zero/ etc. values
        self.current_file = self.processor.files[self.processor.index - 1]
        print(f"In file {self.processor.index - 1}")
        self.episode_number += 1
        
        obs = self.get_observation() # This method returns the current observation, so that for the next step an appropriate action can be taken place
        
        self.done = False if self.svdFailed == False else True # Keeps track of whether we achieved our target or not
        
        return obs, {}
    
    def step(self, action): 
        '''
        This method is called everytime the agent takes an action. We need to tell how to interpret this action in the step() function
        '''
        
        new_node_flag = False # This tells if a new node is added in the circuit
        old_node = None # This keeps track of the old node value which now becomes the new node. Useful for updating the graph of nodes.
        
        # Action unpacking
        comp_type, val_idx, node1, node2 = action
        # print(f"Original Action = [{comp_type}, {val_idx}, {node1}, {node2}]")
        # Is the action valid in the current state or not?
        # This ensures that that there are no floating nodes in the circuit
        
        valid_nodes = len(self.G.nodes) 
        # This gives me the current number of nodes in the circuit as well as the value of the next node. 
        # This also means that the current_node_limit which can be specified by the action space is valid_nodes - 1

        # You can define: current_node_limit = len(valid_nodes)
        if (node1 > valid_nodes - 1) and (node2 > valid_nodes - 1):
            # Invalid node selection (Make them Valid Node)
            node1 = node1 if node1 <= valid_nodes - 1 else valid_nodes - 1
            node2 = node2 if node2 <= valid_nodes - 1 else valid_nodes - 1
        elif (node1 > valid_nodes - 1):
            node1 = valid_nodes - 1

        elif (node2 > valid_nodes - 1):
            node2 = valid_nodes - 1
        elif (node1 == 0 and node2 == 0):
            node1 = 1
            node2 = 1

        
        node2_flag = False # This flag basically tells me if the new node is added to some random (x, x) or to (2, 2).
        
        #-------Case 1--------#
        '''If its added to (2, 2), make that node as (new_node, 2). Change all nodes of the form (x, 2) to (x, new_node)'''
        #-------Case 2--------#
        '''If its added to (x, x), make that node as (x, new_node). Change all nodes of the form (x, y) to (new_node, y)'''
        # If both nodes are same, that means one extra node is required to be added
        if node1 == node2:
            if node1 == 2:
                old_node = node1
                node1 = valid_nodes 
                new_node_flag = True
                node2_flag = True
            else:
                old_node = node2
                node2 = valid_nodes
                new_node_flag = True
                
            
            
        # Handle new node assignment
        if new_node_flag == True:
            self.G.add_node(node1) if node2_flag else self.G.add_node(node2)

        # Add component to circuit        
        # Check if new_node is there. If it is we need to update the entire graph G
        if new_node_flag == True:
            self.G = self.update_Graph(node1 if node2_flag else node2, old_node, node2_flag)
            
        # Till this step update is done. Now to add the new component to the self.components and a new edge to self.G graph.
        self.components.append((comp_type, val_idx, node1, node2))
        self.G.add_edge(node1, node2, type=comp_type, value=val_idx)
        
        
        # print(f"Modified Action = [{comp_type}, {val_idx}, {node1}, {node2}]")
        # Running the simulation and getting the result from LT Spice
        components = [] # This will contain the list to build the circuit
        featureVector = []
        
        for item in self.components:
            components.append([self.component_list[item[0]], self.get_value((item[0]), item[1]), item[2], item[3]])
            

            
        if valid_nodes > 2:
            self.build_circuit(components, "custom_rl_circuit.asc")
            self.run_simulation("custom_rl_circuit.asc", self.LTSpice)
            featureVector, svdFailed = self.plot_response("custom_rl_circuit.raw", "target.csv")

            # -------------------------- # Diagnostic check # -------------------------- 
            def check_invalid(arr, label): 
                arr = np.array(arr, dtype=np.float32) 
                svdFailed = False
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)): 
                    svdFailed = True
                    print(f"⚠️ Invalid values detected in step in {label}: {arr}") 
                return svdFailed 
                
            
            svdFailed = check_invalid(featureVector, "featureVector") 
        obs = self.get_observation() 
        reward = self.get_reward(featureVector, obs, svdFailed) if valid_nodes > 3 else -1
        done = self.is_done() or self.done
        
        print(f"Reward = {reward}")

        return obs, reward, done, False, {}
        
        

    

    def get_observation(self):
        
        
        # Initialize the components array with -1
        obs_components = np.full((self.max_components, 4), -1, dtype=np.float32)

        # Fill in the component details
        for i, (ctype, v_idx, n1, n2) in enumerate(self.components[-self.max_components:]):
            value = self.get_value(ctype, v_idx)  # Extract the component's value
            obs_components[i] = [
                ctype / 2,                              # Normalize type (0,1,2)
                value / (self.value_buckets - 1),        # Normalize value index
                n1 / (self.max_components + 1),          # Normalize node1
                n2 / (self.max_components + 1),          # Normalize node2
            ]

        
        return {
            "components": obs_components,
            "feature_vector": self.featureVector
        }

    def get_value(self, ctype, v_idx):
        return self.bucket_ranges[ctype][v_idx]

    

    def get_reward(self, feature_vector, obs, svdFailed):
        """
        Compute the reward based on circuit parameters and target values.
        Keeps total reward bounded in [-10, +10] to stabilize PPO.
        """
        if svdFailed:
            self.done = True
            return -5.0

        # --------------------------
        # 1. Penalize large circuits
        # --------------------------
        num_components = np.sum(np.any(obs["components"] != -1, axis=1))
        # smaller step penalty so it doesn’t dominate
        reward = -num_components * 0.05  

        # --------------------------
        # 2. Sanitize inputs
        # --------------------------
        def safe_array(x, length=None, default_val=0.0):
            if length == 1:
                x = [x]
            arr = np.array(
                np.nan_to_num(np.array(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            )
            if length is not None:
                if len(arr) < length:
                    arr = np.pad(arr, (0, length - len(arr)), constant_values=default_val)
                elif len(arr) > length:
                    arr = arr[:length]
            return arr

        feature_vector = safe_array(feature_vector, length=86, default_val=0.0)

        # --------------------------
        # 3. Error function (smooth mapping)
        # --------------------------
        def normalized_error(current, target):
            target_achieved = False
            with np.errstate(divide="ignore", invalid="ignore"):
                error = np.abs(current - target) / np.maximum(np.abs(target), 1e-6)
                error = np.clip(error, 0, 10.0)
                error = np.nan_to_num(error, nan=1.0, posinf=1.0, neginf=1.0)

            error_percent = np.clip(np.median(error) * 100, 0, 100)

            
            if error_percent > 30:
                reward_val = -error_percent / 100
            else:
                reward_val = 1 - error_percent/100

            if error_percent < 5:
                reward_val += 5
                target_achieved = True

            return reward_val, target_achieved

        # --------------------------
        # 4. Compute errors
        # --------------------------
        feature_vector_error, feature_vector_target_achieved = normalized_error(feature_vector, obs["feature_vector"])

        self.done = (feature_vector_target_achieved)

        # --------------------------
        # 5. Weighted reward
        # --------------------------
        reward += feature_vector_error

        # --------------------------
        # 6. Clip total reward
        # --------------------------
        reward = float(np.clip(reward, -10.0, 10.0))

        return reward





    def is_done(self):
        if len(self.components) >= self.max_components:
            # print("Max components reached")
            return True
        return False
    

def main():
    env = CircuitEnv(max_components=15, value_buckets=5)

    # Let's define a simple circuit:
    # Component type → 0: Resistor, 1: Inductor, 2: Capacitor
    # v_idx is index in value bucket (0 to value_buckets-1)
    # Nodes n1 and n2 can be arbitrary, let’s use 0, 1, 2 etc.

    # Add a resistor between node 0 and 1
    env.step([0, 5, 0, 1])
    # print(env.components)

    # Add an inductor between node 1 and 2
    env.step([1, 3, 1, 1])
    # print(env.components)

    # Add a capacitor between node 2 and 0
    env.step([2, 7, 1, 2])
    # print(env.components)

    # Print components added
    print("Components (ctype, v_idx, n1, n2):")
    for comp in env.components:
        print(comp)
    print(env.get_observation())
    # print(len(env.components))


if __name__ == "__main__":
    main()