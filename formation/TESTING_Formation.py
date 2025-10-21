import numpy as np

def GenerateBasicFormation():
    """
    Basic 2-1-2 formation (2 defenders, 1 midfielder, 2 forwards)
    This is a balanced formation for general play
    """
    return {
        1: [-14.0, 0.0],    # GK - Goalkeeper
        2: [-10.0, -4.0],   # LD - Left Defender
        3: [-10.0, 4.0],    # RD - Right Defender  
        4: [-6.0, 0.0],     # CM - Center Midfielder
        5: [-2.0, 0.0]      # CF - Center Forward
    }

def GenerateOffensiveFormation():
    """
    1-2-2 formation for offensive pressure
    Pushes more players forward when we have possession
    """
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, -3.0],   # LD - more conservative
        3: [-12.0, 3.0],    # RD - more conservative
        4: [-8.0, -2.0],    # LM - Left Midfielder
        5: [-8.0, 2.0]      # RM - Right Midfielder
    }

def GenerateDefensiveFormation():
    """
    3-1-1 formation for defensive situations
    More players in defensive positions
    """
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, -4.0],   # LCB - Left Center Back
        3: [-12.0, 4.0],    # RCB - Right Center Back
        4: [-10.0, 0.0],    # CB - Center Back
        5: [-6.0, 0.0]      # DM - Defensive Midfielder
    }

def GenerateKickoffFormation():
    """
    Special formation for kickoff situations
    """
    return {
        1: [-14.0, 0.0],    # GK
        2: [-10.0, -4.0],   # LD
        3: [-10.0, 4.0],    # RD
        4: [-2.0, 0.0],     # CF - Kickoff taker
        5: [-6.0, 0.0]      # CM - Support
    }

def GenerateCornerKickOffensiveFormation():
    """
    Formation for when we have a corner kick
    """
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, 0.0],    # LD - Stay back for safety
        3: [-8.0, -4.0],    # RD - Near post
        4: [-5.0, 0.0],     # CM - Edge of box
        5: [-5.0, 3.0]      # CF - Far post
    }

def GenerateCornerKickDefensiveFormation():
    """
    Formation for when opponent has a corner kick
    """
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, -2.0],   # LD - Left post
        3: [-12.0, 2.0],    # RD - Right post
        4: [-10.0, -4.0],   # LCB - Mark left side
        5: [-10.0, 4.0]     # RCB - Mark right side
    }

def GenerateDynamicFormation(ball_position, is_offensive, play_mode=None):
    """
    Generate formation based on ball position, game state, and play mode
    
    Parameters:
    -----------
    ball_position : tuple (x, y)
        Current ball position on field
    is_offensive : bool
        Whether our team is in offensive state
    play_mode : str
        Current game mode
    """
    base_formation = GenerateBasicFormation()
    
    # Adjust based on play mode first
    if play_mode:
        if "KICKOFF" in play_mode:
            if "LEFT" in play_mode:  # Our kickoff
                base_formation = GenerateKickoffFormation()
            else:  # Their kickoff
                base_formation = GenerateDefensiveFormation()
                
        elif "CORNER_KICK" in play_mode:
            if "LEFT" in play_mode:  # Our corner
                base_formation = GenerateCornerKickOffensiveFormation()
            else:  # Their corner
                base_formation = GenerateCornerKickDefensiveFormation()
                
        elif "GOAL_KICK" in play_mode or "FREE_KICK" in play_mode:
            if "LEFT" in play_mode:  # Our set piece
                base_formation = GenerateOffensiveFormation()
            else:  # Their set piece
                base_formation = GenerateDefensiveFormation()
    
    # Adjust based on ball position and offensive/defensive state
    ball_x, ball_y = ball_position
    
    if is_offensive:
        # Use offensive formation but adjust based on ball position
        formation = GenerateOffensiveFormation()
        
        # Push formation forward based on ball position
        ball_influence = max(0, (ball_x + 8) * 0.4)  # Scale factor
        
        for player_num, position in formation.items():
            adjusted_pos = position.copy()
            
            # Different players push forward differently
            if player_num in [4, 5]:  # Midfielders/Forwards
                adjusted_pos[0] += ball_influence * 1.2
            elif player_num in [2, 3]:  # Defenders
                adjusted_pos[0] += ball_influence * 0.6
                
            formation[player_num] = adjusted_pos
            
    else:
        # Defensive formation - adjust based on ball threat
        formation = GenerateDefensiveFormation()
        
        # Pull back based on ball position in our half
        ball_threat = max(0, (-ball_x - 5) * 0.3)  # How deep in our half
        
        for player_num, position in formation.items():
            adjusted_pos = position.copy()
            
            if player_num in [2, 3, 4]:  # Defenders and defensive mid
                adjusted_pos[0] -= ball_threat
            elif player_num == 5:  # Defensive midfielder
                adjusted_pos[0] -= ball_threat * 0.8
                
            formation[player_num] = adjusted_pos
    
    return formation

def GetPlayerRole(player_num, formation_type="basic"):
    """
    Get the role description for a player in a given formation
    """
    role_map = {
        "basic": {
            1: "Goalkeeper",
            2: "Left Defender", 
            3: "Right Defender",
            4: "Center Midfielder",
            5: "Center Forward"
        },
        "offensive": {
            1: "Goalkeeper",
            2: "Left Defender",
            3: "Right Defender", 
            4: "Left Midfielder",
            5: "Right Midfielder"
        },
        "defensive": {
            1: "Goalkeeper",
            2: "Left Center Back",
            3: "Right Center Back",
            4: "Center Back", 
            5: "Defensive Midfielder"
        }
    }
    
    formation_key = formation_type.lower()
    if formation_key not in role_map:
        formation_key = "basic"
        
    return role_map[formation_key].get(player_num, "Unknown")

def ValidateFormation(formation):
    """
    Validate that a formation has valid positions
    """
    if not isinstance(formation, dict):
        return False
        
    if len(formation) != 5:
        return False
        
    for player_num in range(1, 6):
        if player_num not in formation:
            return False
            
        pos = formation[player_num]
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return False
            
        x, y = pos
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
            
        # Check if position is within field bounds (with some tolerance)
        if x < -15 or x > 15 or y < -10 or y > 10:
            return False
            
    return True

# Example usage and testing
if __name__ == "__main__":
    # Test the formations
    basic = GenerateBasicFormation()
    offensive = GenerateOffensiveFormation()
    defensive = GenerateDefensiveFormation()
    dynamic_off = GenerateDynamicFormation((5, 2), True)
    dynamic_def = GenerateDynamicFormation((-8, 1), False)
    
    print("Basic Formation:", basic)
    print("Offensive Formation:", offensive) 
    print("Defensive Formation:", defensive)
    print("Dynamic Offensive:", dynamic_off)
    print("Dynamic Defensive:", dynamic_def)
    
    # Test validation
    print("Basic formation valid:", ValidateFormation(basic))
    print("Offensive formation valid:", ValidateFormation(offensive))