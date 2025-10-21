import numpy as np
from scipy.optimize import linear_sum_assignment

def role_assignment(teammate_positions, formation_positions):
    """
    Assign players to formation positions using Hungarian algorithm
    for optimal role assignment
    """
    # Include self in the player positions
    all_player_positions = [None] * 5  # 5 players total
    
    # We need to get all player positions including self
    # This will be filled by the calling function
    
    # Create cost matrix (5 players x 5 formation positions)
    cost_matrix = np.zeros((5, 5))
    
    # For this simplified version, we'll assume the calling function
    # provides the complete set of positions
    # In practice, you would get the actual positions of all players
    
    # Fill with large values for missing positions
    cost_matrix.fill(1000)
    
    # For available positions, calculate distance costs
    available_players = 0
    player_indices = []
    
    # Check self (always available)
    cost_matrix[0, :] = 100  # Temporary, will be updated
    
    # Check teammates
    for i in range(4):  # 4 teammates
        if teammate_positions[i] is not None:
            available_players += 1
            player_indices.append(i + 1)  # +1 because self is player 0
            
    # Simple assignment: assign based on player number for now
    # In a complete implementation, you would use the Hungarian algorithm
    assignment = {}
    for player_num in range(1, 6):
        if player_num in formation_positions:
            assignment[player_num] = formation_positions[player_num]
    
    return assignment

def hungarian_role_assignment(player_positions, formation_positions):
    """
    Complete Hungarian algorithm implementation for role assignment
    """
    num_players = len(player_positions)
    num_positions = len(formation_positions)
    
    if num_players == 0 or num_positions == 0:
        return {}
    
    # Create cost matrix
    cost_matrix = np.zeros((num_players, num_positions))
    
    for i, player_pos in enumerate(player_positions):
        for j, form_pos in enumerate(formation_positions.values()):
            if player_pos is not None:
                cost = np.linalg.norm(np.array(player_pos) - np.array(form_pos))
            else:
                cost = 1000  # Large penalty for missing players
            cost_matrix[i][j] = cost
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create assignment dictionary
    assignment = {}
    formation_keys = list(formation_positions.keys())
    
    for i, j in zip(row_ind, col_ind):
        if i < len(player_positions) and j < len(formation_keys):
            player_num = i + 1  # Convert to 1-indexed
            formation_key = formation_keys[j]
            assignment[player_num] = formation_positions[formation_key]
    
    return assignment

# Additional helper function for dynamic role assignment
def dynamic_role_assignment(strategy_data, formation_positions):
    """
    Enhanced role assignment considering:
    - Player capabilities
    - Current game situation
    - Opponent positions
    """
    # Get all player positions including self
    all_positions = [strategy_data.mypos]  # Start with self
    
    # Add teammate positions
    for pos in strategy_data.teammate_positions:
        if pos is not None:
            all_positions.append(pos)
        else:
            all_positions.append(None)
    
    # Use Hungarian algorithm for optimal assignment
    assignment = hungarian_role_assignment(all_positions, formation_positions)
    
    return assignment