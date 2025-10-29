import numpy as np

def GenerateDynamicFormation(strategyData):
    ball_x = strategyData.ball_2d[0]
    play_mode = strategyData.play_mode

    # Handle specific game modes first
    if play_mode == 0:  # M_OUR_KICKOFF (KickOff_Left)
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-10, -5]),    # D-L
            np.array([-10, 5]),     # D-R
            np.array([-2, -1]),     # M-L (forward for kickoff)
            np.array([-2, 1])       # M-R (forward for kickoff)
        ]
    elif play_mode == 9:  # M_THEIR_KICKOFF (KickOff_Right)
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-10, -5]),    # D-L
            np.array([-10, 5]),     # D-R
            np.array([-2, -1]),     # A-L (forward attacker)
            np.array([-2, 1])       # A-R (forward attacker)
        ]
    elif play_mode == 4:  # M_OUR_FREE_KICK (freekickLeft)
        # Attacking freekick formation - players spread for options
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-8, -6]),     # D-L (wide option)
            np.array([-8, 6]),      # D-R (wide option)  
            np.array([ball_x + 2, ball_y - 3] if (ball_y := strategyData.ball_2d[1]) > 0 else [ball_x + 2, -3]),  # Near attacker
            np.array([10, 0])       # Far attacker (in goal area)
        ]
    elif play_mode == 13:  # M_THEIR_FREE_KICK (freekickRight)
        # Defensive freekick formation - wall and coverage
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-9, -1.5]),   # D-L (wall left)
            np.array([-9, 1.5]),    # D-R (wall right)
            np.array([-6, -4]),     # M-L (cover left side)
            np.array([-6, 4])       # M-R (cover right side)
        ]
    elif play_mode == 2:  # M_OUR_CORNER_KICK (cornerKickLeft)
        # Attacking corner formation
        ball_y = strategyData.ball_2d[1]
        if ball_y > 0:  # Right side corner
            formation = [
                np.array([-14, 0]),     # GK
                np.array([-12, -4]),    # D-L (back post defender)
                np.array([-8, 6]),      # D-R (near post attacker)
                np.array([-5, 0]),      # M-L (edge of box)
                np.array([-10, 2])      # M-R (far post runner)
            ]
        else:  # Left side corner
            formation = [
                np.array([-14, 0]),     # GK
                np.array([-12, 4]),     # D-L (back post defender)
                np.array([-8, -6]),     # D-R (near post attacker)
                np.array([-5, 0]),      # M-L (edge of box)
                np.array([-10, -2])     # M-R (far post runner)
            ]
    elif play_mode == 11:  # M_THEIR_CORNER_KICK (cornerKickRight)
        # Defensive corner formation
        ball_y = strategyData.ball_2d[1]
        if ball_y > 0:  # Right side corner (their left)
            formation = [
                np.array([-14, 0]),     # GK
                np.array([-12, -1.5]),  # D-L (near post marker)
                np.array([-12, 1.5]),   # D-R (far post marker)
                np.array([-10, -3]),    # M-L (zone defender)
                np.array([-8, 0])       # M-R (edge of box clearance)
            ]
        else:  # Left side corner (their right)
            formation = [
                np.array([-14, 0]),     # GK
                np.array([-12, 1.5]),   # D-L (near post marker)
                np.array([-12, -1.5]),  # D-R (far post marker)
                np.array([-10, 3]),     # M-L (zone defender)
                np.array([-8, 0])       # M-R (edge of box clearance)
            ]
    elif play_mode == 3:  # M_OUR_GOAL_KICK (goalKickLeft)
        # Build from the back formation
        formation = [
            np.array([-14, 0]),     # GK (takes goal kick)
            np.array([-12, -3]),    # D-L (short left option)
            np.array([-12, 3]),     # D-R (short right option)
            np.array([-8, 0]),      # M-L (central midfield option)
            np.array([-5, -2])      # M-R (advanced option)
        ]
    elif play_mode == 12:  # M_THEIR_GOAL_KICK (goalKickRight)
        # Pressing formation for their goal kick
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-10, -4]),    # D-L (high defensive line)
            np.array([-10, 4]),     # D-R (high defensive line)
            np.array([-5, -2]),     # M-L (pressing forward)
            np.array([-5, 2])       # M-R (pressing forward)
        ]
    # Defensive (Ball deep in our half)
    elif ball_x < -5:
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-10, -5]),    # D-L
            np.array([-10, 5]),     # D-R
            np.array([-6, -2]),     # M-L
            np.array([-6, 2])       # M-R
        ]
    # Neutral (Midfield)
    elif -5 <= ball_x <= 3:
        formation = [
            np.array([-12, 0]),
            np.array([-5, -4]),
            np.array([-5, 4]),
            np.array([1, -3]),
            np.array([6, 0])
        ]
    # Offensive (Ball in opponent's half)
    else:
        formation = [
            np.array([-10, 0]),     # GK (sweeper)
            np.array([0, -4]),      # D-L (holding midfield)
            np.array([9, 1]),       # D-R (holding midfield)
            np.array([10, 0]),     # A-L (attacker left)
            np.array([12, 0])       # A-R (attacker right)
        ]
    return formation
