import torch
import torch.nn as nn
import numpy as np

class HeuristicLogits(nn.Module):
    """
    Base class for heuristic logit computation.
    Supported heuristics:
        - linear_time
        - nearest_neighbor
        - nn_plus_one
    """
    def __init__(self, heuristic_type='linear_time'):
        super().__init__()
        self.heuristic_type = heuristic_type

    def compute_heuristic_logits(self, embeddings, state, mat, input):
        """
        Compute heuristic logits based on current state and heuristic type.
        """
        if self.heuristic_type == 'linear_time':
            return self._linear_travel_time_heuristic(embeddings, state, mat, input)
        elif self.heuristic_type == 'nearest_neighbor':
            return self._nearest_neighbor_heuristic(embeddings, state, mat, input)
        elif self.heuristic_type == 'nn_plus_one':
            return self._nn_plus_one_heuristic(embeddings, state, mat, input)
        elif self.heuristic_type == 'nnr':
            return self._nnr_heuristic(embeddings, state, mat, input)
        else:
            raise ValueError(f"Unknown heuristic type: {self.heuristic_type}")

    def _linear_travel_time_heuristic(self, embeddings, state, mat, input):
        """
        Compute logits based on linear travel time from current position.
        Shorter travel time leads to higher logit.
        """
        batch_size = embeddings.size(0)
        graph_size = embeddings.size(1)
        
        current_pos = state.prev_a  # (batch_size, 1)
        _, ind = torch.max(input, dim=2)  # (batch_size, graph_size)
        
        # Initialize travel times tensor
        travel_times = torch.zeros(batch_size, graph_size, device=embeddings.device)
        
        # Compute travel time to each possible destination
        for dest in range(graph_size):
            try:
                # Create destination tensor with proper shape (batch_size, 1)
                dest_tensor = torch.full((batch_size, 1), dest, dtype=torch.long, device=embeddings.device)
                
                # Get travel time from current position to this destination
                travel_time = mat.__getd__(ind, current_pos, dest_tensor, state.lengths)
                
                # Store in the travel_times tensor
                travel_times[:, dest] = travel_time.squeeze()
            except Exception as e:
                print(f"Error computing travel time to destination {dest}: {e}")
                # Set to a default value if there's an error
                travel_times[:, dest] = 1.0
        
        # Add comprehensive safety checks for NaN and inf values
        if torch.isnan(travel_times).any():
            print("NaN in travel_times before fix")
            travel_times = torch.where(torch.isnan(travel_times), torch.ones_like(travel_times), travel_times)
        if torch.isinf(travel_times).any():
            print("Inf in travel_times before fix")
            travel_times = torch.where(torch.isinf(travel_times), torch.ones_like(travel_times), travel_times)
            
        # Ensure all values are finite
        travel_times = torch.where(torch.isfinite(travel_times), travel_times, torch.ones_like(travel_times))
        
        # Convert to logits (negative travel time for softmax preferences)
        heuristic_logits = -travel_times
        
        # Clip extreme values to prevent overflow
        heuristic_logits = torch.clamp(heuristic_logits, min=-10.0, max=10.0)
        
        # Final safety check
        heuristic_logits = torch.where(torch.isfinite(heuristic_logits), heuristic_logits, torch.zeros_like(heuristic_logits))
        
        if torch.isnan(heuristic_logits).any():
            print("NaN in heuristic_logits after processing")
            heuristic_logits = torch.zeros_like(heuristic_logits)
        
        return heuristic_logits

    def _nearest_neighbor_heuristic(self, embeddings, state, mat, input):
        """
        Compute logits based on nearest neighbor strategy using actual travel times.
        Nearest (shortest travel time) nodes get higher logits.
        """
        batch_size = embeddings.size(0)
        graph_size = embeddings.size(1)
        
        current_pos = state.prev_a  # (batch_size, 1)
        _, ind = torch.max(input, dim=2)  # (batch_size, graph_size)
        
        # Initialize travel times tensor
        travel_times = torch.zeros(batch_size, graph_size, device=embeddings.device)
        
        # Compute travel time to each possible destination
        for dest in range(graph_size):
            try:
                dest_tensor = torch.full((batch_size, 1), dest, dtype=torch.long, device=embeddings.device)
                travel_time = mat.__getd__(ind, current_pos, dest_tensor, state.lengths)
                travel_times[:, dest] = travel_time.squeeze()
            except Exception as e:
                print(f"Error computing travel time to destination {dest}: {e}")
                travel_times[:, dest] = 1.0
        
        # Safety checks for NaN and inf
        travel_times = torch.where(torch.isnan(travel_times), torch.ones_like(travel_times), travel_times)
        travel_times = torch.where(torch.isinf(travel_times), torch.ones_like(travel_times), travel_times)
        travel_times = torch.where(torch.isfinite(travel_times), travel_times, torch.ones_like(travel_times))
        
        # Nearest Neighbor logic: negative travel time (shorter time = higher preference)
        # This is essentially the same as linear_time but explicitly named for clarity
        heuristic_logits = -travel_times
        
        # Clip extreme values
        heuristic_logits = torch.clamp(heuristic_logits, min=-10.0, max=10.0)
        
        # Final safety check
        heuristic_logits = torch.where(torch.isfinite(heuristic_logits), heuristic_logits, torch.zeros_like(heuristic_logits))
        
        if torch.isnan(heuristic_logits).any():
            print("NaN in nearest_neighbor heuristic_logits after processing")
            heuristic_logits = torch.zeros_like(heuristic_logits)
        
        return heuristic_logits


    def _nn_plus_one_heuristic(self, embeddings, state, mat, input):
        """
        Compute logits with one-step lookahead (NN+1).
        
        Strategy:
        1. For each candidate next city, compute the travel time to reach it.
        2. Then, from that candidate, compute the travel time to its nearest unvisited neighbor.
        3. Score = travel_time_to_candidate + α * travel_time_from_candidate_to_nearest_unvisited
        4. Convert to logits (negative score, so lower score = higher logit preference).
        
        This captures both immediate cost and future opportunity cost.
        """
        batch_size = embeddings.size(0)
        graph_size = embeddings.size(1)
        
        current_pos = state.prev_a  # (batch_size, 1)
        _, ind = torch.max(input, dim=2)  # (batch_size, graph_size)
        
        # Get the mask of visited nodes
        visited = state.visited_  # (batch_size, 1, graph_size)
        visited = visited.squeeze(1)  # (batch_size, graph_size)
        
        # Initialize scores tensor
        lookahead_scores = torch.zeros(batch_size, graph_size, device=embeddings.device)
        
        # Step 1: Compute travel time from current position to each candidate
        try:
            travel_times_to_candidates = torch.zeros(batch_size, graph_size, device=embeddings.device)
            for candidate in range(graph_size):
                candidate_tensor = torch.full((batch_size, 1), candidate, dtype=torch.long, device=embeddings.device)
                travel_time = mat.__getd__(ind, current_pos, candidate_tensor, state.lengths)
                travel_times_to_candidates[:, candidate] = travel_time.squeeze()
        except Exception as e:
            print(f"Error computing travel time to candidates in NN+1: {e}")
            travel_times_to_candidates = torch.ones(batch_size, graph_size, device=embeddings.device)
        
        # Safety checks
        travel_times_to_candidates = torch.where(torch.isnan(travel_times_to_candidates), 
                                                torch.ones_like(travel_times_to_candidates), 
                                                travel_times_to_candidates)
        travel_times_to_candidates = torch.where(torch.isinf(travel_times_to_candidates), 
                                                torch.ones_like(travel_times_to_candidates), 
                                                travel_times_to_candidates)
        
        # Step 2: For each candidate, find the nearest unvisited node from that candidate
        # and compute the lookahead cost
        lookahead_weight = 0.5  # Weight for lookahead cost (balance immediate vs future)
        
        for candidate in range(graph_size):
            try:
                # Travel time to this candidate from current position
                cost_to_candidate = travel_times_to_candidates[:, candidate]
                
                # Now find minimum travel time from this candidate to any unvisited node
                # (except the candidate itself and current position)
                lookahead_costs = torch.full((batch_size,), float('inf'), device=embeddings.device)
                
                for next_node in range(graph_size):
                    # Skip if this node is already visited or is the candidate itself
                    is_valid = (~visited[:, next_node]) & (torch.arange(graph_size, device=embeddings.device)[next_node] != candidate)
                    
                    if is_valid.any():
                        try:
                            candidate_tensor = torch.full((batch_size, 1), candidate, dtype=torch.long, device=embeddings.device)
                            next_node_tensor = torch.full((batch_size, 1), next_node, dtype=torch.long, device=embeddings.device)
                            
                            # Travel time from candidate to next_node
                            travel_time_from_candidate = mat.__getd__(ind, candidate_tensor, next_node_tensor, state.lengths)
                            travel_time_from_candidate = travel_time_from_candidate.squeeze()
                            
                            # Update lookahead costs: take minimum
                            lookahead_costs = torch.min(lookahead_costs, travel_time_from_candidate)
                        except Exception as e:
                            pass  # Skip if error
                
                # If no valid next node found, set lookahead cost to 0
                lookahead_costs = torch.where(torch.isinf(lookahead_costs), 
                                            torch.zeros_like(lookahead_costs), 
                                            lookahead_costs)
                
                # Combined score: immediate cost + weighted lookahead cost
                combined_score = cost_to_candidate + lookahead_weight * lookahead_costs
                lookahead_scores[:, candidate] = combined_score
                
            except Exception as e:
                print(f"Error computing lookahead for candidate {candidate}: {e}")
                lookahead_scores[:, candidate] = travel_times_to_candidates[:, candidate]
        
        # Safety checks for combined scores
        lookahead_scores = torch.where(torch.isnan(lookahead_scores), 
                                    torch.ones_like(lookahead_scores), 
                                    lookahead_scores)
        lookahead_scores = torch.where(torch.isinf(lookahead_scores), 
                                    torch.ones_like(lookahead_scores), 
                                    lookahead_scores)
        lookahead_scores = torch.where(torch.isfinite(lookahead_scores), 
                                    lookahead_scores, 
                                    torch.ones_like(lookahead_scores))
        
        # Convert scores to logits (negative so lower cost = higher preference)
        heuristic_logits = -lookahead_scores
        
        # Clip extreme values
        heuristic_logits = torch.clamp(heuristic_logits, min=-10.0, max=10.0)
        
        # Final safety check
        heuristic_logits = torch.where(torch.isfinite(heuristic_logits), 
                                    heuristic_logits, 
                                    torch.zeros_like(heuristic_logits))
        
        if torch.isnan(heuristic_logits).any():
            print("NaN in NN+1 heuristic_logits after processing")
            heuristic_logits = torch.zeros_like(heuristic_logits)
        
        return heuristic_logits

    def _nnr_heuristic(self, embeddings, state, mat, input, 
                   top_k=3, probabilities=None):
        """
        Probabilistic Nearest-Neighbor Random (NNR).
        
        Instead of always picking the best, select stochastically from
        the top-k candidates weighted by a probability distribution.
        
        Args:
            embeddings: Node embeddings (batch_size, graph_size, embed_dim)
            state: Current state
            mat: Cost/distance matrix
            input: Input features
            top_k: How many top candidates to consider (default 3)
            probabilities: Distribution over top-k [p1, p2, p3]
                        default [0.8, 0.15, 0.05]
        
        Returns:
            heuristic_logits (batch_size, graph_size) - soft bias toward good choices
        """
        
        if probabilities is None:
            probabilities = torch.tensor([0.8, 0.15, 0.05], device=embeddings.device)
        
        batch_size = embeddings.size(0)
        graph_size = embeddings.size(1)
        current_pos = state.prev_a
        _, ind = torch.max(input, dim=2)
        
        # Compute travel times (same as nearest_neighbor)
        travel_times = torch.zeros(batch_size, graph_size, device=embeddings.device)
        
        for dest in range(graph_size):
            try:
                dest_tensor = torch.full((batch_size, 1), dest, 
                                        dtype=torch.long, device=embeddings.device)
                travel_time = mat.__getd__(ind, current_pos, dest_tensor, state.lengths)
                travel_times[:, dest] = travel_time.squeeze()
            except Exception as e:
                print(f"Error computing travel time to {dest}: {e}")
                travel_times[:, dest] = 1.0
        
        # Safety checks
        travel_times = torch.where(torch.isnan(travel_times), 
                                torch.ones_like(travel_times), 
                                travel_times)
        travel_times = torch.where(torch.isinf(travel_times), 
                                torch.ones_like(travel_times), 
                                travel_times)
        
        # Get top-k candidates (by lowest travel time)
        top_costs, top_indices = torch.topk(travel_times, k=min(top_k, graph_size), 
                                            dim=1, largest=False)
        
        # Assign probabilities to top-k
        # Best gets prob[0], 2nd-best gets prob[1], etc.
        top_probs = probabilities[:top_costs.size(1)]  # Adjust if fewer than k candidates
        
        # Create logits from probabilities (log probabilities as soft bias)
        # Higher probability → higher logit
        heuristic_logits = torch.full((batch_size, graph_size), 
                                    -10.0, device=embeddings.device)
        
        for i in range(min(top_k, graph_size)):
            if i < top_probs.size(0):
                prob_val = top_probs[i]
                # Set logits for top-i nodes
                for b in range(batch_size):
                    if i < top_indices.size(1):
                        node_idx = top_indices[b, i]
                        heuristic_logits[b, node_idx] = torch.log(prob_val + 1e-8)
        
        # Clip and ensure finite
        heuristic_logits = torch.clamp(heuristic_logits, min=-10.0, max=10.0)
        heuristic_logits = torch.where(torch.isfinite(heuristic_logits), 
                                    heuristic_logits, 
                                    torch.zeros_like(heuristic_logits))
        
        return heuristic_logits




# Add more helper classes/functions if needed below
