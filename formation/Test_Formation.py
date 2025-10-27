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
    elif -5 <= ball_x <= 8:
        formation = [
            np.array([-12, 0]),
            np.array([-5, -4]),
            np.array([-5, 4]),
            np.array([1, -2]),
            np.array([1, 2])
        ]

    # Offensive (Ball in opponent’s half)
    else:
        # Compress attackers closer to goal for faster shooting
        formation = [
            np.array([-10, 0]),     # GK (forward sweep)
            np.array([4, -4]),      # D-L (advanced)
            np.array([4, 4]),       # D-R (advanced)
            np.array([12, -1.5]),   # F-L (in box)
            np.array([12, 1.5])     # F-R (in box)
        ]
    return formation
