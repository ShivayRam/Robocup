import numpy as np

#this is da basic formation
def GenerateBasicFormation():

    #For general play, balanced with (2 def, 1 mid, 2 fwd)

    return {
        1: [-14, 0],     #GK
        2: [-10, -4],    #RCB
        3: [-10, 4],     #LCB
        4: [-6, 0],      #CM
        5: [-2, 0]       #CF
    }


#offensive formation
def GenerateOffensiveFormation():

    #offensive pressure, more players up front with (1 def, 2 mid, 2 fwd)

    return {
        1: [-14, 0],    #GK
        2: [-12, -3],   #LCB
        3: [-12, 3],    #RCB
        4: [-8, -2],    #LM
        5: [-8, 2]      #RM
    }

#Defensive formation
def GenerateDefensiveFormation():

    #defensive situatrions, sit deep with (3 def, 1 mid, 1 fwd)

    return {
        1: [-14, 0],    #GK
        2: [-10, -4],   #RCB
        3: [-10, 4],    #LCB
        4: [-12, 0],    #CB
        5: [-6, 0]      #CM

    }

#kickoff formation
def GenerateKickoffFormation():

    #fomration for kickoff situations

    return {
        1: [-14, 0],    #GK
        2: [-10, -4],   #RCB
        3: [-10, 4],    #LCB
        4: [-6, 0],     #CM
        5: [-2, 0]      #CF

    }

#corner kick Attacking
def GenerateCornerKickOffensiveFormation():

    #formation when taking corner kick
    #Not sure if I got these positions right. Might need to adjust later.
    return {
        1: [-14, 0],
        2: [-10, 0],
        3: [8, 4],
        4: [8, -4],
        5: [10, 0]
    }


#corner kick Defending
def GenerateCornerKickDefensiveFormation():

    #formation when defending against corner
    return {
        1: [-14, 0],
        2: [-12, -2],
        3: [-12, 2],
        4: [-10, -4],
        5: [-10, 4]

    }

#next gonna try create a dynamic formation
#based on ball position and our teammates positions
def GenerateDynamicFormation(ball_position, is_offensive, play_mode=None):

    #storing the basic formation in var
    base_form = GenerateBasicFormation()

    #have to adjust based on play mode first
    if play_mode:

        if "KICKOFF" in play_mode:

            #check if our kickoff
            if "LEFT" in play_mode:

                base_form = GenerateKickoffFormation()
            
            else: #their kickoff

                base_form = GenerateDefensiveFormation()
        
        elif "CORNER_KICK" in play_mode:

            if "LEFT" in play_mode: #our corner

                base_form = GenerateCornerKickOffensiveFormation()

            else: #their corner
            
                base_form = GenerateCornerKickDefensiveFormation()
            
        elif "GOAL_KICK" in play_mode or "FREE_KICK" in play_mode:

            if "LEFT" in play_mode: #our ball

                base_form = GenerateOffensiveFormation()

            else: #their ball

                base_form = GenerateDefensiveFormation()

    
    #based on ball pos and off/def form, adjust
    ball_x, ball_y = ball_position

    if is_offensive:

        #use da offesnive form but adjust depending on ball pos
        formation = GenerateOffensiveFormation()

        #push fwd based on ball pos
        ball_influence = max(0, (ball_x +8) * 0.4) #scale factor

        for player_num, position in formation.items():

            adjusted_pos = position.copy()

            #defenders and fwds push different amounts
            if player_num in [4, 5]:
                adjusted_pos[0] += ball_influence * 1.2

            elif player_num in [2, 3]:
                adjusted_pos[0] += ball_influence * 0.6

            formation[player_num] = adjusted_pos

    
    else:

        #def position
        formation = GenerateDefensiveFormation()

        #get back based on the ball pos in ur half
        ball_threat = max(0, (-ball_x -5) * 0.3) #how deep in our half MIGHT CHANGE

        for player_num, position in formation.items():

            adjusted_pos = position.copy()

            #def and DM drops back more
            if player_num in [2, 3, 4]:
                adjusted_pos[0] -= ball_threat

            elif player_num == 5:
                adjusted_pos[0] -= ball_threat * 0.8

            formation[player_num] = adjusted_pos

    return formation


#gets da role description based on their number in a function
def GetPlayerRole(player_number, formation_type="basic"):

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


    #Gives function that retuin more than 1 predefined formations and includes gettiong player roles
    formation_key = formation_type.lower()

    if formation_key not in role_map:

        formation_key = "basic"

    return role_map[formation_key].get(player_number, "Unknown")


#validates the formation structure and ensures bots dont go off pitch
def ValidateFormation(formation):

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