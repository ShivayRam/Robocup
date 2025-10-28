import numpy as np

def GenerateDynamicFormation(strategyData):
    ball_x = strategyData.ball_2d[0]

    # Defensive (Ball deep in our half)
    if ball_x < -5:
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-10, -5]),    # D-L
            np.array([-10, 5]),     # D-R
            np.array([-6, -2]),     # M-L
            np.array([-6, 2])       # M-R
        ]

    # Neutral (Midfield)
    elif -5 <= ball_x <= 6:
        formation = [
            np.array([-12, 0]),
            np.array([-5, -2]),
            np.array([-5, 2]),
            np.array([1, -1]),
            np.array([6, 1])
        ]

    # Offensive (Ball in opponent's half) - MODIFIED: 3 attackers up front
    else:
        formation = [
            np.array([-10, 0]),     # GK (sweeper)
            np.array([0, 0]),      # D-L (holding midfield)
            np.array([10, 0]),       # D-R (holding midfield)
            np.array([13, -1]),     # A-L (attacker left)
            np.array([13, 1])       # A-R (attacker right)
        ]
    return formation