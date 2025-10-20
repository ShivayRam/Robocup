
import math


#first im gonna make a function for preference list based on the distance
def calc_pref_list(player_pos, formation_positions):

    """
    so using a player's pos (x,y) and a list of formation positions,
    we compute da sorted list of indices of formation pos in ascending order
    euclid dist
    """
    dist = []
    px, py = player_pos

    
    for index, (fx, fy) in enumerate(formation_positions):
        
        
        d = math.hypot(fx - px, fy - py)
        
        dist.append((d, index))
    
    #now sort by the distance and then by da indaex
    dist.sort(key = lambda item: (item[0], item[1]))

    #now return the indices only
    return [index for (_, index) in dist]

#this function is for gale shapley stable marrige
def g_s_stable(player_prefs, role_prefs):

    """
    Gale-Shaply for stable marriage. return a dictionary of player indexes to their matched role indexes

    """

    #at first the players are free duh
    free_pl = list(player_prefs.keys())

    #nobody is matched at first
    player_matched = {p: None for p in player_prefs}
    role_matched   = {r: None for r in role_prefs}

    #now we track next proposal index for each of the players
    nxt_prop_index = {p: 0 for p in player_prefs}

    #ranking for roles. each role maps player to their rank
    role_ranking = {}
    for r, prefs in role_prefs.items():

        rank_map = {p: ind for ind, p in enumerate(prefs)}
        
        role_ranking[r] = rank_map

    while free_pl:

        p = free_pl.pop(0)

        #which of the roles p proposes next
        r = player_prefs[p][ nxt_prop_index[p] ]
        nxt_prop_index[p] += 1

        curr_p_for_r = role_matched[r]

        if curr_p_for_r is None:

            #role r is free → match p and r
            player_matched[p] = r

            role_matched[r] = p
        
        else:

            

            #check if role prefers p over player_2
            if role_ranking[r][p] < role_ranking[r][curr_p_for_r]:

                #if yes then switch
                player_matched[p] = r
                role_matched[r] = p

                #player 2 becomes single/free
                player_matched[curr_p_for_r] = None

                free_pl.append(curr_p_for_r)

            else:
                #if doesnt prefer player p
                free_pl.append(p)

    return player_matched

def role_assignment(teammate_positions, formation_positions): 

    # Input : Locations of all teammate locations and positions
    # Output : Map from unum -> positions
    #-----------------------------------------------------------#


    #compute da pref lists for da players
    n = len(teammate_positions)

    #build the pref lists o da playas
    player_prefs = {p_ind: calc_pref_list(teammate_positions[p_ind], formation_positions)

                    for p_ind in range(n)}

    #build da pref lists fo da roles
    role_prefs = {}

    for r_ind in range(n):

        role_prefs[r_ind] = calc_pref_list(formation_positions[r_ind], teammate_positions)

    #do the stable marriaging
    match = g_s_stable(player_prefs, role_prefs)

    #conv matching into the right format output
    point_preferences = {}

    for p_ind, r_ind in match.items():

        unum = p_ind + 1

        assigned_pos = formation_positions[r_ind]
        point_preferences[unum] = assigned_pos

    return point_preferences