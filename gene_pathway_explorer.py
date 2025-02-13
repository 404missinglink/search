import random
from datasets import load_dataset
from collections import defaultdict
import re
from typing import List, Dict, Set, Optional, Tuple
import math
import networkx as nx
import matplotlib.pyplot as plt

class GeneNode:
    def __init__(self, gene_data: dict):
        self.gene_data = gene_data
        self.visits = 0
        self.score = 0
        self.children: Dict[str, 'GeneNode'] = {}  # Map of gene_symbol -> GeneNode
        self.parent: Optional['GeneNode'] = None
        self.gene_symbol = gene_data['gene_symbol']
        
    @property
    def location(self) -> str:
        return str(self.gene_data['location'])
    
    @property
    def chromosome(self) -> Optional[str]:
        if not self.location:
            return None
        match = re.match(r'(\d+|X|Y|MT)', self.location)
        return match.group(1) if match else None
    
    @property
    def gene_groups(self) -> Set[str]:
        groups = self.gene_data['gene_group']
        if isinstance(groups, str):
            return {g.strip() for g in groups.split('|')}
        return set()
    
    @property
    def chromosome_band(self) -> Optional[str]:
        """Extract the chromosome band (e.g., p36.13 from 1p36.13)"""
        if not self.location:
            return None
        match = re.match(r'\d+([pq]\d+\.?\d*)', self.location)
        return match.group(1) if match else None
    
    def ucb1_score(self, total_visits: int, exploration_weight: float = 1.414) -> float:
        """Calculate UCB1 score for node selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.score / self.visits
        exploration = exploration_weight * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration

class GenePathwayExplorer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.gene_data = {row['gene_symbol']: row for row in dataset['train']}
        self.build_gene_groups()
        self.explored_edges = set()
        
    def build_gene_groups(self):
        self.gene_group_map = defaultdict(set)
        self.chromosome_band_map = defaultdict(set)
        
        for gene_symbol, data in self.gene_data.items():
            node = GeneNode(data)
            # Build gene group mapping
            for group in node.gene_groups:
                self.gene_group_map[group].add(gene_symbol)
            
            # Build chromosome band mapping
            if node.chromosome and node.chromosome_band:
                self.chromosome_band_map[f"{node.chromosome}{node.chromosome_band}"].add(gene_symbol)
    
    def calculate_band_distance(self, band1: str, band2: str) -> float:
        """Calculate a rough distance score between chromosome bands"""
        if not (band1 and band2):
            return float('inf')
        
        # Same band
        if band1 == band2:
            return 0
        
        # Different arms (p/q)
        if band1[0] != band2[0]:
            return 100
        
        # Same arm, different numbers
        try:
            num1 = float(band1[1:])
            num2 = float(band2[1:])
            return abs(num1 - num2)
        except:
            return 50
    
    def get_possible_moves(self, node: GeneNode, visited: Set[str], target_gene: str) -> List[Tuple[str, float]]:
        target_node = GeneNode(self.gene_data[target_gene])
        possible_moves = []
        
        for gene_symbol, data in self.gene_data.items():
            if gene_symbol in visited:
                continue
            
            candidate = GeneNode(data)
            score = 0.0
            
            # Proximity on same chromosome
            if candidate.chromosome == node.chromosome:
                band_distance = self.calculate_band_distance(
                    node.chromosome_band,
                    candidate.chromosome_band
                )
                score += max(0, 50 - band_distance)
            
            # Shared gene groups (functional similarity)
            shared_groups = len(candidate.gene_groups & node.gene_groups)
            score += shared_groups * 20
            
            # Bonus for moving towards target chromosome
            if candidate.chromosome == target_node.chromosome:
                score += 30
            
            if score > 0:
                possible_moves.append((gene_symbol, score))
        
        return sorted(possible_moves, key=lambda x: x[1], reverse=True)[:5]  # Top 5 candidates
    
    def select_node(self, root: GeneNode, visited: Set[str]) -> GeneNode:
        """Select a promising node using UCB1"""
        node = root
        total_visits = root.visits
        
        while node.children:
            best_score = float('-inf')
            best_child = None
            
            for child in node.children.values():
                if child.gene_symbol in visited:
                    continue
                score = child.ucb1_score(total_visits)
                if score > best_score:
                    best_score = score
                    best_child = child
            
            if not best_child:
                break
                
            node = best_child
            visited.add(node.gene_symbol)
        
        return node
    
    def expand_node(self, node: GeneNode, visited: Set[str], target_gene: str) -> Optional[GeneNode]:
        """Expand node by adding a new child"""
        possible_moves = self.get_possible_moves(node, visited, target_gene)
        
        for gene_symbol, _ in possible_moves:
            if gene_symbol not in node.children and gene_symbol not in visited:
                child = GeneNode(self.gene_data[gene_symbol])
                child.parent = node
                node.children[gene_symbol] = child
                self.explored_edges.add((node.gene_symbol, gene_symbol))
                return child
        
        return None
    
    def simulate(self, node: GeneNode, visited: Set[str], target_gene: str, max_steps: int = 5) -> float:
        """Simulate a random path from node to target"""
        current = node
        path = [current.gene_symbol]
        current_visited = visited.copy()
        
        for _ in range(max_steps):
            if current.gene_symbol == target_gene:
                return 100.0 / len(path)
            
            moves = self.get_possible_moves(current, current_visited, target_gene)
            if not moves:
                break
            
            # Weighted random choice
            total_score = sum(score for _, score in moves)
            if total_score == 0:
                break
            
            rand = random.uniform(0, total_score)
            cumsum = 0
            chosen_gene = moves[0][0]  # Default to first move
            
            for gene, score in moves:
                cumsum += score
                if cumsum >= rand:
                    chosen_gene = gene
                    break
            
            current = GeneNode(self.gene_data[chosen_gene])
            path.append(current.gene_symbol)
            current_visited.add(current.gene_symbol)
        
        return self.score_path(path, target_gene)
    
    def backpropagate(self, node: GeneNode, score: float):
        """Update statistics for all nodes in the path"""
        while node:
            node.visits += 1
            node.score += score
            node = node.parent
    
    def score_path(self, path: List[str], target_gene: str) -> float:
        if not path:
            return 0.0
            
        if path[-1] == target_gene:
            return 100.0 / len(path)  # Reward shorter successful paths
            
        current = GeneNode(self.gene_data[path[-1]])
        target = GeneNode(self.gene_data[target_gene])
        
        score = 0.0
        
        # Chromosome proximity
        if current.chromosome == target.chromosome:
            band_distance = self.calculate_band_distance(
                current.chromosome_band,
                target.chromosome_band
            )
            score += max(0, 50 - band_distance)
        
        # Functional similarity through shared groups
        shared_groups = len(current.gene_groups & target.gene_groups)
        score += shared_groups * 20
        
        # Path length penalty
        return score / len(path)
    
    def find_path(self, start_gene: str, target_gene: str, iterations: int = 1000) -> Tuple[List[str], float]:
        """Find path using Monte Carlo Tree Search"""
        root = GeneNode(self.gene_data[start_gene])
        best_path = None
        best_score = float('-inf')
        
        for _ in range(iterations):
            visited = {start_gene}
            
            # Selection
            selected_node = self.select_node(root, visited)
            
            # Expansion
            new_node = self.expand_node(selected_node, visited, target_gene)
            if new_node:
                selected_node = new_node
                visited.add(selected_node.gene_symbol)
            
            # Simulation
            score = self.simulate(selected_node, visited, target_gene)
            
            # Backpropagation
            self.backpropagate(selected_node, score)
            
            # Track best path
            current = selected_node
            path = []
            while current:
                path.append(current.gene_symbol)
                current = current.parent
            path.reverse()
            
            path_score = self.score_path(path, target_gene)
            if path_score > best_score:
                best_score = path_score
                best_path = path
        
        return best_path, best_score
    
    def visualize_pathway(self, path: List[str], output_file: str = "gene_pathway.png"):
        G = nx.Graph()
        
        # Add all explored edges
        for source, target in self.explored_edges:
            G.add_edge(source, target)
        
        # Prepare node colors based on chromosomes and sizes based on gene groups
        chromosome_colors = {}
        node_colors = []
        node_sizes = []
        
        for node_id in G.nodes():
            # Create GeneNode for each node
            node = GeneNode(self.gene_data[node_id])
            
            # Color by chromosome
            chrom = node.chromosome
            if chrom not in chromosome_colors:
                chromosome_colors[chrom] = plt.cm.Set3(len(chromosome_colors) / 12.)
            node_colors.append(chromosome_colors[chrom])
            
            # Size by number of gene groups
            num_groups = len(node.gene_groups)
            node_sizes.append(1000 + num_groups * 200)
        
        # Prepare edge colors - highlight the final path
        edge_colors = []
        edge_widths = []
        path_edges = list(zip(path[:-1], path[1:]))
        
        for edge in G.edges():
            if edge in path_edges or (edge[1], edge[0]) in path_edges:
                edge_colors.append('red')
                edge_widths.append(2.0)
            else:
                edge_colors.append('lightgray')
                edge_widths.append(0.5)
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1.5, iterations=50)
        
        # Draw the network
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.6)
        
        # Add labels with chromosome bands
        labels = {node_id: f"{node_id}\n({GeneNode(self.gene_data[node_id]).location})" 
                 for node_id in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Add a title
        plt.title("Gene Pathway Network\nNode size indicates number of functional groups\nRed edges show the optimal path between genes")
        
        # Add chromosome legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color, label=f'Chromosome {chrom}',
                                    markersize=10)
                         for chrom, color in chromosome_colors.items()]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    # Load dataset
    ds = load_dataset("dwb2023/hgnc_gene_mapping")
    explorer = GenePathwayExplorer(ds)
    
    # Get available genes
    available_genes = list(explorer.gene_data.keys())
    print(f"Total number of genes in dataset: {len(available_genes)}")
    
    # Try to find genes on different chromosomes for more interesting paths
    genes_by_chrom = defaultdict(list)
    for gene in available_genes:
        node = GeneNode(explorer.gene_data[gene])
        if node.chromosome:
            genes_by_chrom[node.chromosome].append(gene)
    
    # Find chromosomes with the most genes
    top_chroms = sorted(genes_by_chrom.items(), key=lambda x: len(x[1]), reverse=True)[:2]
    
    if len(top_chroms) >= 2:
        start_gene = random.choice(top_chroms[0][1])
        target_gene = random.choice(top_chroms[1][1])
        start_node = GeneNode(explorer.gene_data[start_gene])
        target_node = GeneNode(explorer.gene_data[target_gene])
        print(f"\nExploring gene pathway between:")
        print(f"Start: {start_gene} (Chromosome {start_node.chromosome}, Location: {start_node.location})")
        print(f"Target: {target_gene} (Chromosome {target_node.chromosome}, Location: {target_node.location})")
    else:
        start_gene = available_genes[0]
        target_gene = available_genes[-1]
    
    print("\nSearching for pathway that optimizes:")
    print("1. Chromosomal proximity (genes closer on chromosomes)")
    print("2. Functional relationships (shared gene groups)")
    print("3. Path efficiency (shorter paths preferred)")
    
    path, score = explorer.find_path(start_gene, target_gene)
    
    if not path:
        print("No path found between the genes.")
        return
        
    print(f"\nDiscovered gene pathway (score: {score:.2f}):")
    for i, gene in enumerate(path):
        node = GeneNode(explorer.gene_data[gene])
        print(f"{i+1}. {gene}")
        print(f"   Location: {node.location}")
        if node.gene_groups:
            print(f"   Functional Groups: {', '.join(node.gene_groups)}")
        else:
            print("   Functional Groups: None")
        
        # Show relationship to next gene if not last
        if i < len(path) - 1:
            next_node = GeneNode(explorer.gene_data[path[i+1]])
            if node.chromosome == next_node.chromosome:
                band_distance = explorer.calculate_band_distance(
                    node.chromosome_band,
                    next_node.chromosome_band
                )
                print(f"   → Distance to next gene: {band_distance:.2f} bands")
            shared = len(node.gene_groups & next_node.gene_groups)
            if shared:
                print(f"   → Shared groups with next: {shared}")
        print()
    
    # Generate visualization
    print("\nGenerating network visualization...")
    explorer.visualize_pathway(path)
    print("Visualization saved as 'gene_pathway.png'")

if __name__ == "__main__":
    main() 