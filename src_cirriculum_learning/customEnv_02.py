import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import csvProcessor_01 as rd
import extract_features_05 as fe
import subprocess
import os
import csv


good_circuits = []


class CircuitEnv(gym.Env):
    def __init__(self, max_components=3, value_buckets=5,
                 folder_path="../3_data", work_dir="./ltspice_env_0"):
        super(CircuitEnv, self).__init__()

        # --- File management ---
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.asc_file = os.path.join(self.work_dir, "circuit.asc")
        self.raw_file = os.path.join(self.work_dir, "circuit.raw")
        self.csv_file = os.path.join(self.work_dir, "response.csv")
        self.counter_file = os.path.join(self.work_dir, "simulation_counter.txt")

        self.simulation_counter = 0

        # --- Env parameters ---
        self.max_components = max_components
        self.value_buckets = value_buckets
        self.node_counter = 2  # 0 = GND, 1 = VDD
        self.episode_number = 0
        self.pole_count = 6
        self.feature_vector_size = 6 * self.pole_count + 2
        # Value buckets for components
        self.bucket_ranges = {
            0: np.array([1000, 2000, 3000, 4000, 5000]),   # R
            2: np.array([0.1, 0.5, 1.0, 2.0, 2.5]),        # L
            1: np.array([3e-6, 2e-6, 1e-6, 4e-6, 5e-6])    # C
        }
        self.component_list = ["res", "cap", "ind"]

        # --- Action / observation spaces ---
        self.action_space = spaces.MultiDiscrete([3, value_buckets, max_components, max_components])
        self.observation_space = spaces.Dict({
            "components": spaces.Box(-1, 1, shape=(self.max_components, 4), dtype=np.float32),
            "feature_vector": spaces.Box(-np.inf, np.inf, shape=(self.feature_vector_size,), dtype=np.float32),
        })

        # --- LTSpice executable ---
        self.LTSpice = r"C:\Users\jishu\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

        # CSVProcessor for targets
        self.processor = rd.CSVProcessor(folder_path)
        
        # Persistent log file (always append here)
        self.log_file = os.path.join(self.work_dir, "target.csv")
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Step", "Reward", "CurrentFile"])
                
        # Step/episode counters
        self.current_step = 0
        self.episode_id = 0

        self.reset()

    # ------------------ Graph updates ------------------
    def update_Graph(self, new_node_value, old_node_value, node2_flag):
        new_components = []
        for item in self.components:
            if node2_flag:
                node_value = new_node_value if item[3] == old_node_value else item[3]
                new_components.append((item[0], item[1], item[2], node_value))
            else:
                node_value = new_node_value if item[2] == old_node_value else item[2]
                new_components.append((item[0], item[1], node_value, item[3]))

        self.components = new_components
        new_graph = nx.DiGraph()
        new_graph.add_nodes_from([0, 1])
        for u, v, data in self.G.edges(data=True):
            edge_type, edge_value = data['type'], data['value']
            if v == old_node_value:
                v = new_node_value
            new_graph.add_edge(u, v, type=edge_type, value=edge_value)
        return new_graph

    # ------------------ LTSpice integration ------------------
    def build_circuit(self, component_list, filename=None):
        if filename is None:
            filename = self.asc_file

        nodes = set()
        for comp in component_list:
            n1, n2 = str(comp[2]), str(comp[3])
            if n1 != '0': nodes.add(n1)
            if n2 != '0': nodes.add(n2)

        sorted_nodes = sorted(nodes, key=int)
        node_index_map = {node: idx for idx, node in enumerate(sorted_nodes)}

        wires, component_blocks, wire_blocks, flag_blocks = [], [], [], []
        delta = 0
        counts = {"res": 0, "cap": 0, "ind": 0, "voltage": 0}

        for idx in range(len(sorted_nodes)):
            x = idx * 112
            wires.append(f"WIRE {x} -168 {x} {(len(component_list)) * 80}")

        for comp in component_list:
            ctype, value, node1, node2 = comp
            node1, node2 = str(node1), str(node2)
            counts[ctype] = counts.get(ctype, 0) + 1
            inst_name = f"{ctype[0].upper()}{counts[ctype]}"

            if node1 != '0' and node2 != '0':
                idx1, idx2 = node_index_map[node1], node_index_map[node2]
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
            ])

            if node1 == '0' or node2 == '0':
                offset = 0 if ctype == "cap" else 16
                flag_blocks.append(f"FLAG {x - offset} {y + 16} 0")
            else:
                x1 = x - (0 if ctype == "cap" else 16)
                x2 = max(idx1, idx2) * 112
                wire_blocks.append(f"WIRE {x1} {y+16} {x2} {y+16}")

        with open(filename, 'w') as f:
            f.write("Version 4\nSHEET 1 880 680\n")
            f.write("SYMBOL voltage -272 -176 R0\n")
            f.write("SYMATTR InstName AC1\n")
            f.write("SYMATTR Value AC 1\n")
            f.write("FLAG -272 -80 0\n")
            f.write("WIRE 0 -160 -272 -160\n")
            f.write("TEXT -380 328 Left 2 !.ac dec 100 1 30k\n")

            if len(sorted_nodes) > 2:
                last_node = sorted_nodes[-1]
                idx_last = node_index_map[last_node]
                x = idx_last * 112
                flag_blocks.append(f"FLAG {x} {-144} Vout")

            if len(sorted_nodes) > 1:
                y = (len(component_list)) * 80
                flag_blocks.append(f"FLAG {112} {y} 0")

            for line in wires + component_blocks + wire_blocks + flag_blocks:
                f.write(line + "\n")

    def run_simulation(self, circuit_path, ltspice_path):
        self.simulation_counter += 1
        with open(self.counter_file, "w") as f:
            f.write(str(self.simulation_counter))

        command = [ltspice_path, "-b", "-run", circuit_path]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error running LTSpice: {e}")
            return None

    def plot_response(self, raw_file_path=None, output_csv=None):
        from PyLTSpice import RawRead
        import csv

        raw_file_path = raw_file_path or self.raw_file
        output_csv = output_csv or self.csv_file

        ltr = RawRead(raw_file_path)
        trace_names = ltr.get_trace_names()
        freq = ltr.get_trace("frequency")
        f_vals = np.real(freq.get_wave(0))

        if "V(vout)" in trace_names:
            vout = ltr.get_trace("V(vout)")
            vout_vals = vout.get_wave(0)
            gain = np.abs(vout_vals)
            phase_rad = np.angle(vout_vals)

            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Frequency", "Gain", "Phase"])
                for f, g, p in zip(f_vals, gain, phase_rad):
                    writer.writerow([f, g, p])

            featureVector, svdFailed = fe.extract_feature_vector(file_path=output_csv)
        else:
            featureVector, svdFailed = [], True

        return featureVector, svdFailed

    # ------------------ RL interface ------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.components = []
        self.G = nx.DiGraph()
        self.G.add_nodes_from([0, 1])
        self.node_counter = 2

        self.featureVector, self.svdFailed = self.processor.process_next()
        self.current_file = self.processor.files[self.processor.index - 1]
        self.episode_number += 1
        self.done = False if not self.svdFailed else True
        
        self.episode_id += 1
        self.current_step = 0

        return self.get_observation(), {}

    def step(self, action):
        new_node_flag = False
        old_node = None

        comp_type, val_idx, node1, node2 = action
        valid_nodes = len(self.G.nodes)

        # Fix invalid nodes
        if (node1 > valid_nodes - 1) and (node2 > valid_nodes - 1):
            node1, node2 = valid_nodes - 1, valid_nodes - 1
        elif node1 > valid_nodes - 1:
            node1 = valid_nodes - 1
        elif node2 > valid_nodes - 1:
            node2 = valid_nodes - 1
        elif node1 == 0 and node2 == 0:
            node1, node2 = 1, 1

        node2_flag = False
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

        if new_node_flag:
            self.G.add_node(node1 if node2_flag else node2)

        if new_node_flag:
            self.G = self.update_Graph(node1 if node2_flag else node2, old_node, node2_flag)

        self.components.append((comp_type, val_idx, node1, node2))
        self.G.add_edge(node1, node2, type=comp_type, value=val_idx)

        components = []
        for item in self.components:
            components.append([
                self.component_list[item[0]],
                self.get_value(item[0], item[1]),
                item[2], item[3]
            ])

        featureVector, svdFailed = [], True
        if valid_nodes > 2:
            self.build_circuit(components, self.asc_file)
            self.run_simulation(self.asc_file, self.LTSpice)
            featureVector, svdFailed = self.plot_response(self.raw_file, self.csv_file)

        obs = self.get_observation()
        reward = self.get_reward(featureVector, obs, svdFailed) if valid_nodes > 3 else -1
        done = self.is_done() or self.done
        
        self.current_step += 1
        
        # Current file being processed
        current_file_name = f"target_{self.processor.index - 1}"

        # Append log entry
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.episode_id, self.current_step, reward, current_file_name])

        return obs, reward, done, False, {}

    def get_observation(self):
        obs_components = np.full((self.max_components, 4), -1, dtype=np.float32)
        for i, (ctype, v_idx, n1, n2) in enumerate(self.components[-self.max_components:]):
            value = self.get_value(ctype, v_idx)
            obs_components[i] = [
                ctype / 2,
                value / (self.value_buckets - 1),
                n1 / (self.max_components + 1),
                n2 / (self.max_components + 1),
            ]
        return {"components": obs_components, "feature_vector": self.featureVector}

    def get_value(self, ctype, v_idx):
        return self.bucket_ranges[ctype][v_idx]

    # def get_reward(self, feature_vector, obs, svdFailed):
    #     if svdFailed:
    #         self.done = True
    #         return -5.0

    #     num_components = np.sum(np.any(obs["components"] != -1, axis=1))
    #     reward = -num_components * 0.05

    #     def safe_array(x, length=None, default_val=0.0):
    #         arr = np.array(np.nan_to_num(np.array(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    #         if length and len(arr) < length:
    #             arr = np.pad(arr, (0, length - len(arr)), constant_values=default_val)
    #         elif length and len(arr) > length:
    #             arr = arr[:length]
    #         return arr

    #     feature_vector = safe_array(feature_vector, length=self.feature_vector_size, default_val=0.0)

    #     def normalized_error(current, target):
    #         with np.errstate(divide="ignore", invalid="ignore"):
    #             error = np.abs(current - target) / np.maximum(np.abs(target), 1e-6)
    #             error = np.clip(error, 0, 10.0)
    #             error = np.nan_to_num(error, nan=1.0, posinf=1.0, neginf=1.0)
    #         error_percent = np.clip(np.median(error) * 100, 0, 100)
    #         reward_val = 1 - error_percent / 100 if error_percent <= 30 else -error_percent / 100
    #         target_achieved = error_percent < 5
    #         if target_achieved:
    #             reward_val += 5
    #         return reward_val, target_achieved

    #     feature_vector_error, feature_vector_target_achieved = normalized_error(feature_vector, obs["feature_vector"])
    #     self.done = feature_vector_target_achieved
    #     reward += feature_vector_error
    #     return float(np.clip(reward, -10.0, 10.0))
    
    def get_reward(self, feature_vector, obs, svdFailed):
        if svdFailed:
            self.done = True
            return -5.0

        num_components = np.sum(np.any(obs["components"] != -1, axis=1))
        reward = -num_components * 0.05

        similarity_value = fe.normalized_circuit_similarity(feature_vector, self.featureVector)
        if similarity_value < 0.5:
            reward -= 1 - similarity_value
        elif similarity_value > 0.9:
            reward += 5
            self.done = True
        else: 
            reward += 0.5

        return float(np.clip(reward, -10.0, 10.0))

    def is_done(self):
        return len(self.components) >= self.max_components
