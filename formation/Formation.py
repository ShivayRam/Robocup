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

