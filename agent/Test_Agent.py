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

        # 💡 MODIFICATION 1: Initial position now matches the Neutral Zone formation for consistency.
        # This is the formation used when the ball is at (0, 0)
        self.init_pos = (
            np.array([-12, 0]),     # GK 
            np.array([-5, -4]),     # D-L
            np.array([-5, 4]),      # D-R
            np.array([0, -2]),      # M-L 
            np.array([0, 2])        # M-R
        )[unum-1]


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
            # 💡 MODIFICATION 2: Explicitly beam to initial position before kickoff
            if strategyData.play_mode == self.world.M_BEFORE_KICKOFF:
                self.beam(avoid_center_circle=True) 
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

    def select_skill(self, strategyData):
            drawer = self.world.draw
            
            # 1. Team Strategy: Dynamic Formation and Role Assignment
            
            # Use the dynamic formation based on ball position
            formation_positions = GenerateDynamicFormation(strategyData)
            
            # Use Gale-Shapley (Submission 1) to assign player to role
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
            
            # Update player's desired position and orientation
            strategyData.my_desired_position = point_preferences[strategyData.player_unum]
            
            # All non-active players should face the ball. Active player will face the goal.
            if strategyData.active_player_unum != strategyData.robot_model.unum:
                strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d)
            else:
                # Active player faces the opponent's goal (15, 0)
                strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(np.array([15, 0]))

            drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")
            
            # 2. Primitive Actions Decision Tree (If-Else Structure)
            
            # --- A. Active Player Logic ---
            if strategyData.active_player_unum == strategyData.robot_model.unum: 
                drawer.annotation((0, 10.5), "Active Player: Ball Control", drawer.Color.yellow, "status")
                
                # Sub-Decision: Shoot, Pass, or Dribble?
                
                # 1. Check for immediate shot opportunity (e.g., within 8m of the goal)
                OPPONENT_GOAL_CENTER = np.array([15, 0])
                dist_to_goal = np.linalg.norm(np.array(strategyData.mypos) - OPPONENT_GOAL_CENTER)
                
                if dist_to_goal < 8.0:
                    
                    # Target the Goal Corners (15, 1.05) and (15, -1.05)
                    if np.random.rand() > 0.5:
                        shoot_target = np.array([15, 1.05]) # Top corner
                        corner_text = "TOP CORNER"
                    else:
                        shoot_target = np.array([15, -1.05]) # Bottom corner
                        corner_text = "BOTTOM CORNER"
                    
                    # Shoot at the chosen target
                    drawer.annotation(strategyData.mypos, "SHOOT! (" + corner_text + ")", drawer.Color.red, "action_text")
                    return self.kickTarget(strategyData, strategyData.mypos, shoot_target)
                
                # 2. Otherwise, dribble forward (i.e., kick to a spot in front of current pos)
                else:
                    # Calculate a dribble target 2 meters in front of the current position, towards the goal (15, 0)
                    dribble_vec = (OPPONENT_GOAL_CENTER - np.array(strategyData.mypos))
                    dribble_vec = dribble_vec / np.linalg.norm(dribble_vec) * 2.0 # Normalize and extend by 2m
                    dribble_target = np.array(strategyData.mypos) + dribble_vec
                    
                    drawer.annotation(strategyData.mypos, "DRIBBLE", drawer.Color.green, "action_text")
                    return self.kickTarget(strategyData, strategyData.mypos, dribble_target)


            # --- B. Support Player Logic ---
            else:
                drawer.clear("status")
                drawer.clear("action_text")
                
                # 1. If player is not at their role position, move there.
                # Using IsFormationReady() logic for the individual player:
                teammate_pos = strategyData.teammate_positions[strategyData.player_unum - 1]
                desired_pos = strategyData.my_desired_position
                
                # Use a threshold for 'ready' position (e.g., within 0.5m)
                if teammate_pos is None or np.linalg.norm(teammate_pos - desired_pos) > 0.5:
                    # Move to position
                    drawer.annotation(strategyData.mypos, "MOVING", drawer.Color.cyan, "action_text")
                    return self.move(desired_pos, orientation=strategyData.my_desired_orientation)
                else:
                    # Hold position and face the ball
                    drawer.annotation(strategyData.mypos, "HOLD", drawer.Color.blue, "action_text")
                    # Return move with distance_to_final_target=0 to stop walking animation and just face the ball
                    return self.move(desired_pos, orientation=strategyData.my_desired_orientation)