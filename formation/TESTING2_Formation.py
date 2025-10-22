import numpy as np

def GenerateBasicFormation():
    """Basic 2-1-2 formation for general play"""
    return {
        1: [-14.0, 0.0],    # GK
        2: [-10.0, -4.0],   # Right Defender
        3: [-10.0, 4.0],    # Left Defender  
        4: [-6.0, 0.0],     # Center Midfielder
        5: [-2.0, 0.0]      # Center Forward
    }

def GenerateOffensiveFormation():
    """1-2-2 formation for offensive pressure"""
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, -3.0],   # Right Defender
        3: [-12.0, 3.0],    # Left Defender
        4: [-8.0, -2.0],    # Left Midfielder
        5: [-8.0, 2.0]      # Right Midfielder
    }

def GenerateDefensiveFormation():
    """3-1-1 formation for defensive situations"""
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, -4.0],   # Right Center Back
        3: [-12.0, 4.0],    # Left Center Back
        4: [-10.0, 0.0],    # Center Back
        5: [-6.0, 0.0]      # Defensive Midfielder
    }

def GenerateKickoffFormation():
    """Formation for kickoff situations"""
    return {
        1: [-14.0, 0.0],    # GK
        2: [-10.0, -4.0],   # Right Defender
        3: [-10.0, 4.0],    # Left Defender
        4: [-6.0, 0.0],     # Center Midfielder
        5: [-2.0, 0.0]      # Center Forward (kickoff taker)
    }

def GenerateCornerKickOffensiveFormation():
    """Formation when taking corner kick - with players in opponent half"""
    return {
        1: [-14.0, 0.0],    # GK stays in goal
        2: [-10.0, 0.0],    # Defender stays back
        3: [8.0, 4.0],      # Attacker - near post
        4: [8.0, -4.0],     # Attacker - far post
        5: [10.0, 0.0]      # Attacker - center
    }

def GenerateCornerKickDefensiveFormation():
    """Formation when defending against corner"""
    return {
        1: [-14.0, 0.0],    # GK
        2: [-12.0, -2.0],   # Left post
        3: [-12.0, 2.0],    # Right post
        4: [-10.0, -4.0],   # Left defender
        5: [-10.0, 4.0]     # Right defender
    }

def GenerateDynamicFormation(ball_position, is_offensive, play_mode=None):
    """Generate formation based on ball position and game state"""
    # Use appropriate formation based on play mode
    if play_mode:
        if "KICKOFF" in play_mode:
            if "LEFT" in play_mode:  # Our kickoff
                formation = GenerateKickoffFormation()
            else:  # Their kickoff
                formation = GenerateDefensiveFormation()
        elif "CORNER_KICK" in play_mode:
            if "LEFT" in play_mode:  # Our corner
                formation = GenerateCornerKickOffensiveFormation()
            else:  # Their corner
                formation = GenerateCornerKickDefensiveFormation()
        elif "GOAL_KICK" in play_mode or "FREE_KICK" in play_mode:
            if "LEFT" in play_mode:  # Our set piece
                formation = GenerateOffensiveFormation()
            else:  # Their set piece
                formation = GenerateDefensiveFormation()
        else:
            # For other play modes (including PlayOn), use basic formation
            formation = GenerateBasicFormation()
    else:
        formation = GenerateBasicFormation()

    # Adjust the formation only during PlayOn
    if play_mode == "PlayOn":
        ball_x, ball_y = ball_position
        if is_offensive:
            # Use offensive formation and adjust based on ball position
            ball_influence = max(0, (ball_x + 8) * 0.4)  # Scale factor
            for player_num, position in formation.items():
                adjusted_pos = position.copy()
                if player_num in [4, 5]:  # Midfielders and Forwards
                    adjusted_pos[0] += ball_influence * 1.2
                elif player_num in [2, 3]:  # Defenders
                    adjusted_pos[0] += ball_influence * 0.6
                formation[player_num] = adjusted_pos
        else:
            # Defensive formation - adjust based on ball threat
            ball_threat = max(0, (-ball_x - 5) * 0.3)  # How deep in our half
            for player_num, position in formation.items():
                adjusted_pos = position.copy()
                if player_num in [2, 3, 4]:  # Defenders and defensive mid
                    adjusted_pos[0] -= ball_threat
                elif player_num == 5:  # Defensive midfielder
                    adjusted_pos[0] -= ball_threat * 0.8
                formation[player_num] = adjusted_pos

    return formation

def GetPlayerRole(player_number, formation_type="basic"):
    """Get role description for player"""
    role_map = {
        "basic": {
            1: "Goalkeeper",
            2: "Right Defender",
            3: "Left Defender",
            4: "Center Midfielder", 
            5: "Center Forward"
        },
        "offensive": {
            1: "Goalkeeper",
            2: "Right Defender",
            3: "Left Defender",
            4: "Left Midfielder",
            5: "Right Midfielder"
        },
        "defensive": {
            1: "Goalkeeper", 
            2: "Right Center Back",
            3: "Left Center Back",
            4: "Center Back",
            5: "Defensive Midfielder"
        }
    }
    
    formation_key = formation_type.lower()
    if formation_key not in role_map:
        formation_key = "basic"
        
    return role_map[formation_key].get(player_number, "Unknown")

def ValidateFormation(formation):
    """Validate formation structure"""
    if not isinstance(formation, dict):
        return False
    
    if len(formation) != 5:
        return False
    
    for player_number in range(1, 6):
        if player_number not in formation:
            return False
        
        pos = formation[player_number]
        if not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return False
        
        x, y = pos
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False
            
    return True