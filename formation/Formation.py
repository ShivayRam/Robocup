import numpy as np

"""def GenerateBasicFormation():


    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-7, -2]),  # Left Defender
        np.array([-0, 3]),   # Right Defender
        np.array([7, 1]),    # Forward Left
        np.array([12, 0])      # Forward Right
    ]



    # formation = [
    #     np.array([-13, 0]),    # Goalkeeper
    #     np.array([-10, -2]),  # Left Defender
    #     np.array([-11, 3]),   # Center Back Left
    #     np.array([-8, 0]),    # Center Back Right
    #     np.array([-3, 0]),   # Right Defender
    #     np.array([0, 1]),    # Left Midfielder
    #     np.array([2, 0]),    # Center Midfielder Left
    #     np.array([3, 3]),     # Center Midfielder Right
    #     np.array([8, 0]),     # Right Midfielder
    #     np.array([9, 1]),    # Forward Left
    #     np.array([12, 0])      # Forward Right
    # ]

    return formation
"""


#Gonna make a custom dynamic formation function
def GenerateDynamicFormation(strategyData):

    """
        This function is going to change the formation of the bots
        depending on the game mode but also where the ball is on the field
        depending on its x coordinate
    """

    #assigned the game mode and ball x coordinate to vars
    play_mode = strategyData.play_mode
    ball_x = strategyData.ball_2d[0]

    #start but checking game mode currently at play and adjust
    
    #our kickoff (Kickoff_Left)
    if play_mode == 0:
        formation = [

            np.array([-14, 0]),     # GK
            np.array([-8, -3]),    #RCB
            np.array([-8, 3]),     #LCB
            np.array([-2, -1]),     #RCF
            np.array([-2, 1])       #LCF
        ]

    #their kickoff (Kickoff_Right)
    elif play_mode == 9:

        formation = [

            np.array([-14, 0]),     # GK
            np.array([-10, -3]),    #RCB
            np.array([-10, 3]),     #LCB
            np.array([-2, -1]),     #RCF
            np.array([-2, 1])       #LCF
        ]

    #our Freekick (Freekick_Left)
    elif play_mode == 4:

        formation = [

            np.array([-14, 0]),    #GK
            np.array([-8, 0]),    #CB
            np.array([-4, 0]),     #CDM
            np.array([4, 0]),     #CM
            np.array([10, 0])      #ST
        ]

    #their Freekick (Freekick_Right)
    elif play_mode == 13:

        formation = [

            np.array([-14, 0]),    #GK
            np.array([-9, -1]),    #CB
            np.array([-9, 1]),     #CB
            np.array([-6, 0]),     #CDM
            np.array([5, 0])      #ST
        ]

    #our Fcornerkick (CornerKick_Left)
    elif play_mode == 2:

        #check if left or right sided corner
        ball_y = strategyData.ball_2d[1]

        if ball_y > 0: #left sided corner
            formation = [

                np.array([-14, 0]),    #GK
                np.array([-5, 0]),    #CB
                np.array([3, 0]),     #CM
                np.array([12, 3]),     #LCF
                np.array([13, 3])      #RCF
            ]
        
        else: #right sided corner

            formation = [

                np.array([-14, 0]),    #GK
                np.array([-5, 0]),    #CB
                np.array([3, 0]),     #CM
                np.array([12, -3]),     #LCF
                np.array([13, -3])      #RCF
            ]
    
    #their Cornerkick (CornerKick_Right)
    elif play_mode == 11:

        #check if left or right sided corner
        ball_y = strategyData.ball_2d[1]

        if ball_y > 0: #left sided corner
            formation = [

                np.array([-14, 0]),     #GK
                np.array([-11, 1]),     #CB
                np.array([-11, 0]),     #CB
                np.array([-5, 1]),      #CM
                np.array([0, 0])        #ST
            ]
        
        else: #right sided corner

            formation = [

                np.array([-14, 0]),     #GK
                np.array([-11, 0]),     #CB
                np.array([-11, -1]),    #CB
                np.array([-5, -1]),     #CM
                np.array([0, 0])        #ST
            ]

    #our goalkick (GoalKick_Left)
    elif play_mode == 3:

         formation = [

                np.array([-14, 0]),     #GK
                np.array([-12, -1]),    #CB
                np.array([-12, 1]),     #CB
                np.array([-8, 0]),      #CM
                np.array([3, 0])        #ST
            ]
         
    #their goalkick (GoalKick_Right)
    elif play_mode == 12:

         formation = [

                np.array([-14, 0]),     #GK
                np.array([-5, 0]),      #CB
                np.array([0, 0]),       #CM
                np.array([8, 0]),       #RCF
                np.array([10, 2])       #LCF
            ]
         
    
    #The following are for the normal play mode (Play_On)


    #deep in our half
    elif play_mode == 20 and ball_x < -5:

        formation = [

            np.array([-14, 0]),     #GK
            np.array([-10, -2]),    #RCB
            np.array([-10, 2]),     #LCB
            np.array([-5, 0]),      #CM
            np.array([-2, 0])       #ST 
        ]

    #middle of the pitch
    elif play_mode == 20 and -5 <= ball_x <= 2:

        formation = [

            np.array([-12, 0]),     #GK
            np.array([-5, -2]),    #RCB
            np.array([-5, 2]),     #LCB
            np.array([5, 0]),      #CM
            np.array([8, 0])       #ST 
        ]

    
    #final third
    elif play_mode == 20 and ball_x > 3:

        formation = [

            np.array([-10, 0]),     #GK
            np.array([-2, 0]),      #CB
            np.array([7, 1]),       #CDM
            np.array([10, 0]),      #CAM
            np.array([12, 0])       #ST 
        ]
    
    else:

        #by standing formation
        formation = [
            np.array([-14, 0]),     #GK
            np.array([-10, -1]),    #CB
            np.array([-10, 1]),     #CB
            np.array([-6, 0]),      #CM
            np.array([-3, 0])       #ST

        ]

    return formation