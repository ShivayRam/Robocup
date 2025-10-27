import numpy as np

def GenerateDynamicFormation(strategyData):
    ball_x = strategyData.ball_2d[0]
    ball_y = strategyData.ball_2d[1]

    # Defensive (ball deep in our half)
    if ball_x < -5:
        formation = [
            np.array([-14, 0]),     # GK
            np.array([-10, -4]),    # D-L
            np.array([-10, 4]),     # D-R
            np.array([-6, -2]),     # M-L
            np.array([-6, 2])       # M-R
        ]

    # Midfield (ball in neutral zone)
    elif -5 <= ball_x <= 8:
        formation = [
            np.array([-12, 0]),
            np.array([-3, -2]),
            np.array([-3, 2]),
            np.array([2, -1]),
            np.array([2, 1])
        ]

    # Offensive (ball in opponent’s half) — compact attack
    else:
        formation = [
            np.array([-10, 0]),       # GK
            np.array([5, -1]),        # D-L (advanced)
            np.array([5, 1]),         # D-R (advanced)
            np.array([12, -1]),       # F-L (support ahead)
            np.array([11, 1])         # F-R (support ahead)
        ]

    return formation
