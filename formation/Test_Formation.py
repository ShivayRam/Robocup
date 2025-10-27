import numpy as np
# from strategy.Strategy import Strategy # You might need to pass Strategy object to this function

def GenerateDynamicFormation(strategyData):
    # Ball X-coordinate is the primary driver for shifting the formation
    ball_x = strategyData.ball_2d[0]
    
    # Define zones based on the field layout (Field X goes from -15 to 15)
    
    # Zone 1: Defensive (Ball in our half, X < -5)
    if ball_x < -5:
        # Puts most players deep to defend the goal at (-15, 0)
        formation = [
            np.array([-14, 0]),     # GK (Deep)
            np.array([-10, -5]),    # D-L
            np.array([-10, 5]),     # D-R
            np.array([-5, -2]),     # M-L
            np.array([-5, 2])       # M-R (Midfield support in our half)
        ]
        
    # Zone 2: Neutral (Ball in midfield, -5 <= X <= 5)
    elif -5 <= ball_x <= 5:
        # Balanced formation
        formation = [
            np.array([-12, 0]),     # GK (Slightly up)
            np.array([-5, -4]),     # D-L
            np.array([-5, 4]),      # D-R
            np.array([0, -2]),      # M-L (Center-field)
            np.array([0, 2])        # M-R (Center-field)
        ]
        
    # Zone 3: Offensive (Ball in opponent's half, X > 5)
    else: # ball_x > 5
        # Pushes players forward to support the attack on the goal at (15, 0)
        formation = [
            np.array([-10, 0]),     # GK (Pushed up to receive a clearance)
            np.array([2, -5]),      # D-L (Pushed into opponent's half)
            np.array([2, 5]),       # D-R (Pushed into opponent's half)
            np.array([10, -2]),     # F-L (Attacker support)
            np.array([10, 2])       # F-R (Attacker support)
        ]
        
    return formation