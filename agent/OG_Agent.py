from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 

from formation.Formation import GenerateDynamicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation (optional)
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()

    def think_and_send(self):
        
        behavior = self.behavior
        strategyData = Strategy(self.world)
        d = self.world.draw

        if strategyData.play_mode == self.world.M_GAME_OVER:
            pass
        elif strategyData.PM_GROUP == self.world.MG_ACTIVE_BEAM:
            self.beam()
        elif strategyData.PM_GROUP == self.world.MG_PASSIVE_BEAM:
            self.beam(True) # avoid center circle
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            
            if strategyData.play_mode == self.world.M_BEFORE_KICKOFF:
                self.beam(True) # avoid center circle
            
            elif strategyData.play_mode == self.world.M_OUR_KICKOFF:
                self.handle_our_kickoff(strategyData)
            
            elif strategyData.play_mode == self.world.M_THEIR_KICKOFF:
                self.handle_their_kickoff(strategyData)

            elif strategyData.play_mode == self.world.M_OUR_FREE_KICK:
                self.handle_our_free_kick(strategyData)

            elif strategyData.play_mode == self.world.M_THEIR_FREE_KICK:
                self.handle_their_free_kick(strategyData)
            
            elif strategyData.play_mode == self.world.M_OUR_GOAL_KICK:
                self.handle_our_goal_kick(strategyData)

            elif strategyData.play_mode == self.world.M_THEIR_GOAL_KICK:
                self.handle_their_goal_kick(strategyData)
            
            elif strategyData.play_mode == self.world.M_OUR_CORNER_KICK:
                self.handle_our_corner_kick(strategyData)
            
            elif strategyData.play_mode == self.world.M_THEIR_CORNER_KICK:
                self.handle_their_corner_kick(strategyData)

            else:
                self.select_skill(strategyData)
                


        #--------------------------------------- 3. Broadcast
        self.radio.broadcast()

        #--------------------------------------- 4. Send to server
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""


#function to handle our kickoff(duh)
    def handle_our_kickoff(self, strategyData):

        drawer = self.world.draw
        drawer.annotation((0, 11), "Our Kickoff - ATTACK MODE" , drawer.Color.green, "game_mode")
        

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]

        if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
            drawer.annotation(strategyData.mypos, "KICKOFF TAKER", drawer.Color.yellow, "action_text")
            return self.kickTarget(strategyData,strategyData.mypos,np.array([15,0])) # Kicks to Opponents Goal
        
        else:

            return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))
        

#now the function to handle their kickoff
    def handle_their_kickoff(self, strategyData):

        drawer = self.world.draw
        drawer.annotation((0, 11), "THEIR KICKOFF - DEFENSIVE", drawer.Color.orange, "game_mode")
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]

        if strategyData.player_unum in [4, 5]:
            drawer.annotation(strategyData.mypos, "HOLD DEFENSIVE LINE", drawer.Color.blue, "action_text")

        else:
            drawer.annotation(strategyData.mypos, "DEFENSIVE POSITON", drawer.Color.blue, "action_text")

        return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))
    

#function to handle when its our freekick

    def handle_our_free_kick(self, strategyData):

        drawer = self.world.draw
        drawer.annotation((0, 11), "OUR FREEKICK - ATTACK SETUP", drawer.Color.green, "game_mode")

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]

        ball_pos = np.array(strategyData.ball_2d)

        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation(strategyData.mypos, "FREEKICK TAKER", drawer.Color.yellow, "action_text")

            #define opponent goal
            OPPONENT_GOAL = np.array([15,0])
            #euclidean dist to goal from ball position
            dist_to_goal = np.linalg.norm(ball_pos - OPPONENT_GOAL)

            if ball_pos[0] < -5:

                best_target = None
                min_dist = float('inf')

                for unum in [4, 5]:

                    teammate_pos = strategyData.teammate_positions[unum - 1]

                    if teammate_pos is not None:
                        
                        dist = np.linalg.norm(ball_pos = teammate_pos)
                        if dist < min_dist:

                            min_dist = dist
                            best_target = teammate_pos

                if best_target is not None:

                    drawer.annotation(strategyData.mypos, "PASS TO ATTACKER", drawer.Color.cyan, "action_text")
                    return self.kickTarget(strategyData,strategyData.mypos,best_target)
                
                else:
                    drawer.annotation(strategyData.mypos, "CLEAR UP FIELD", drawer.Color.cyan, "action_text")
                    return self.kickTarget(strategyData,strategyData.mypos,OPPONENT_GOAL)
                
            
            elif dist_to_goal < 15:
            
                if abs(ball_pos[1]) < 3:

                    shoot_target = np.array([15, 0])
            
                else:

                    #15, -1.5 part of the goal if the ball is on positive y side, else 1.5
                    shoot_target = np.array([15, -1.5 if ball_pos[1] > 0 else 1.5])

                drawer.annotation(strategyData.mypos, "DIRECT SHOT", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData,strategyData.mypos,shoot_target)
        

            else:

                best_target = None
                max_advance = -20

                for i, teammate_pos in enumerate(strategyData.teammate_positions):

                    if teammate_pos is not None and i+1 != strategyData.player_unum:

                        if teammate_pos[0] > max_advance:

                            max_advance = teammate_pos[0]
                            best_target = teammate_pos
            

                if best_target is not None:

                    drawer.annotation(strategyData.mypos, "STRATEGIC PASS", drawer.Color.cyan, "action_text")
                    return self.kickTarget(strategyData,strategyData.mypos,best_target)
            
                else:

                    advance_target = ball_pos + (OPPONENT_GOAL - ball_pos) * 0.3
                    drawer.annotation(strategyData.mypos, "ADVANCE BALL", drawer.Color.green, "action_text")
                    return self.kickTarget(strategyData,strategyData.mypos,advance_target)
            

        else:

            if strategyData.player_unum in [4, 5]:

                current_pos = np.array(strategyData.mypos)
                desired_pos = strategyData.my_desired_position

                if np.linalg.norm(current_pos - desired_pos) < 1:

                    run_target = desired_pos + np.array([3, 0]) # move 3 meters forward
                    run_target[0] = min(run_target[0], 13) # do not go beyond x=13

                    drawer.annotation(strategyData.mypos, "MAKE RUN", drawer.Color.magenta, "action_text")
                    return self.move(run_target, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))
                
            
            drawer.annotation(strategyData.mypos, "HOLD POSITION", drawer.Color.blue, "action_text")

            return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))
        

#function to handle behaviour when their freekick
    def handle_their_free_kick(self, strategyData):

        #drawer = self.world.draw

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        red_position = point_preferences[strategyData.player_unum]

        ball_pos = np.array(strategyData.ball_2d)

        if strategyData.player_unum in [2, 3]:

            desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos)
            return self.move(red_position, orientation=desired_orientation)
        
        elif strategyData.player_unum == 1:

            ball_to_goal_vec = np.array([-14,0]) - ball_pos
            ball_to_goal_vec = ball_to_goal_vec / np.linalg.norm(ball_to_goal_vec)

            if ball_pos[1] > 0:
                gk_pos = np.array([-14, -0.5])

            else:
                gk_pos = np.array([-14, 0.5])

            return self.move(gk_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
        
        else:
            closest_opponent = None
            min_dist = 5.0

            for i, opp_pos in enumerate(strategyData.opponent_positions):

                if opp_pos is not None:

                    dist = np.linalg.norm(np.array(strategyData.mypos) - opp_pos)

                    if dist < min_dist:

                        min_dist = dist
                        closest_opponent = opp_pos
            
            if closest_opponent is not None:

                mark_pos = closest_opponent - (closest_opponent - ball_pos) * 0.3
                return self.move(mark_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            
            else:

                return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            

    #function to handle our corner kicks
    def handle_our_corner(self, strategyData):

        drawer = self.world.draw

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]


        ball_pos = np.array(strategyData.ball_2d)
        ball_y = ball_pos[1]

        if strategyData.active_player_unum == strategyData.robot_model.unum:

            if ball_y > 0:

                cross_target = np.array([-8, 3])
            else:
                cross_target = np.array([-8, -3])

            return self.kickTarget(strategyData,strategyData.mypos,cross_target)
        
        else:

            if strategyData.player_unum == 1:

                drawer.annotation(strategyData.mypos, "GOALKEEPER - STAY READY", drawer.Color.blue, "action_text")

            elif strategyData.player_unum in [2, 3]:

                current_pos = np.array(strategyData.mypos)

                if np.linalg.norm(current_pos - strategyData.my_desired_position) < 1:

                    run_target = strategyData.my_desired_position + np.array([2, 0])

                    return self.move(run_target, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))
                else:
                    drawer.annotation(strategyData.mypos, "HOLD POS", drawer.Color.blue, "action_text")

            else:

                drawer.annotation(strategyData.mypos, "ATTACKING POS", drawer.Color.blue, "action_text")

            return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
        

#and obviously the func to handle their corners

    def handle_their_corner(self, strategyData):

        drawer = self.world.draw

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]


        ball_pos = np.array(strategyData.ball_2d)

        if strategyData.player_unum == 1:

            drawer.annotation(strategyData.mypos, "GOALKEEPER - CORNER DEF", drawer.Color.red, "action_text")

            if ball_pos[1] > 0:
                gk_pos = np.array([-14, -1])

            else:
                gk_pos = np.array([-14, 1])

            return self.move(gk_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
        
        elif strategyData.player_unum in [2, 3]:

            return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
        
        else:
            closest_opponent = None
            min_dist = 4.0

            for i, opp_pos in enumerate(strategyData.opponent_positions):

                if opp_pos is not None:

                    if -12 <= opp_pos[0] <= -8 and -5 <= opp_pos[1] <= 5:

                        dist = np.linalg.norm(np.array(strategyData.mypos) - opp_pos)

                        if dist < min_dist:

                            min_dist = dist
                            closest_opponent = opp_pos

            if closest_opponent is not None:

                mark_pos = closest_opponent + (closest_opponent - ball_pos) * 0.1
                return self.move(mark_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            
            else:
                return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            

#functions to handle our and their goal kicks

    def handle_our_goalkick(self, strategyData):

        drawer = self.world.draw

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]


        ball_pos = np.array(strategyData.ball_2d)

        if strategyData.player_unum == 1:

            #drawer.annotation(strategyData.mypos, "GOALKEEPER - GOAL KICK", drawer.Color.yellow, "action_text")

            best_target = None
            min_opponent_pressure = float('inf')


            for i, teammate_pos in enumerate(strategyData.teammate_positions):

                if teammate_pos is not None and 1+1 != 1:

                    opponent_pressure = 0

                    for opp_pos in strategyData.opponent_positions:

                        if opp_pos is not None:

                            distance_to_teammate = np.linalg.norm(opp_pos - teammate_pos)

                            if distance_to_teammate < 3:

                                opponent_pressure += (3 - distance_to_teammate)

                    if opponent_pressure < min_opponent_pressure:

                        min_opponent_pressure = opponent_pressure
                        best_target = teammate_pos

            if best_target is not None and min_opponent_pressure < 2:
                return self.kickTarget(strategyData,strategyData.mypos,best_target)
            
            else:
                clear_target = np.array([0, 0])
                return self.kickTarget(strategyData,strategyData.mypos,clear_target)
            
        
        else:
            return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
        

    def handle_their_goalkick(self, strategyData):

        drawer = self.world.draw

        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]


        ball_pos = np.array(strategyData.ball_2d)

        if strategyData.player_unum in [4, 5]:

            #drawer.annotation(strategyData.mypos, "GOALKEEPER - DEFEND GOAL KICK", drawer.Color.red, "action_text")

            closest_opponent = None
            min_dist = 6

            for i, opp_pos in enumerate(strategyData.opponent_positions):

                if opp_pos is not None and opp_pos[0] > -10:

                    dist = np.linalg.norm(np.array(strategyData.mypos) - opp_pos)

                    if dist < min_dist:

                        min_dist = dist
                        closest_opponent = opp_pos

            if closest_opponent is not None and min_dist < 5:
                
                press_pos = closest_opponent + (closest_opponent - ball_pos) * 0.2
                return self.move(press_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            
        else:
            return self.move(strategyData.my_desired_position, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))





    def select_skill(self,strategyData):
        #--------------------------------------- 2. Decide action
        drawer = self.world.draw
        path_draw_options = self.path_manager.draw_options


        #------------------------------------------------------
        #Role Assignment
        if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
            drawer.annotation((0,10.5), "Role Assignment Phase" , drawer.Color.yellow, "status")
        else:
            drawer.clear("status")

        formation_positions = GenerateDynamicFormation()
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.my_desired_position)

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2,drawer.Color.blue,"target line")

        if not strategyData.IsFormationReady(point_preferences):
            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desried_orientation)
        #else:
        #     return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)


    
        #------------------------------------------------------
        # Example Behaviour
        target = (15,0) # Opponents Goal

        if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
            drawer.annotation((0,10.5), "Pass Selector Phase" , drawer.Color.yellow, "status")
        else:
            drawer.clear_player()

        if strategyData.active_player_unum == strategyData.robot_model.unum: # I am the active player 
            pass_reciever_unum = strategyData.player_unum + 1 # This starts indexing at 1, therefore player 1 wants to pass to player 2
            if pass_reciever_unum != 6:
                target = strategyData.teammate_positions[pass_reciever_unum-1] # This is 0 indexed so we actually need to minus 1 
            else:
                target = (15,0) 

            drawer.line(strategyData.mypos, target, 2,drawer.Color.red,"pass line")
            return self.kickTarget(strategyData,strategyData.mypos,target)
        else:
            drawer.clear("pass line")
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)
        

