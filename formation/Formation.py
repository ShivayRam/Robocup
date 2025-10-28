
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
            np.array([-5, 0]),
            np.array([-3, 0]),
            np.array([1, -1]),
            np.array([8, 1])
        ]
    # Offensive (Ball in opponent's half)
    else:
        formation = [
            np.array([-10, 0]),     # GK (sweeper)
            np.array([0, -4]),      # D-L (holding midfield)
            np.array([10, 0]),       # D-R (holding midfield)
            np.array([11, 0]),     # A-L (attacker left)
            np.array([13, 1])       # A-R (attacker right)
        ]
    return formation