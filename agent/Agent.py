
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

        # FIX: Initialize init_pos properly
        self.init_pos = (
            np.array([-12, 0]),     # GK 
            np.array([-5, -4]),     # D-L
            np.array([-5, 4]),      # D-R
            np.array([0, -2]),      # M-L 
            np.array([0, 2])        # M-R
        )[unum-1]
        
        # Cache for faster decision making
        self.last_decision_time = 0
        self.cached_formation = None
        self.cached_point_prefs = None
        self.last_play_mode = None

    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        # FIX: Use the actual init_pos attribute
        pos = self.init_pos.copy() # copy the position array
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
            # Handle different game modes
            if strategyData.play_mode == self.world.M_BEFORE_KICKOFF:
                self.beam(avoid_center_circle=True) 
            elif strategyData.play_mode == self.world.M_OUR_KICKOFF:
                self.handle_our_kickoff(strategyData)
            elif strategyData.play_mode == self.world.M_THEIR_KICKOFF:
                self.handle_their_kickoff(strategyData)
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

    def handle_our_kickoff(self, strategyData):
        """Special handling for our kickoff"""
        drawer = self.world.draw
        drawer.annotation((0, 11), "OUR KICKOFF - ATTACK MODE", drawer.Color.green, "game_mode")
        
        # Use formation-based positioning but with kickoff logic
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        # Active player (closest to ball) handles kickoff
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation(strategyData.mypos, "KICKOFF TAKER", drawer.Color.yellow, "action_text")
            # Kick toward opponent goal
            return self.kickTarget(strategyData, strategyData.mypos, np.array([10, 0]))
        else:
            # Support players move to kickoff formation positions
            return self.move(strategyData.my_desired_position, 
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))

    def handle_their_kickoff(self, strategyData):
        """Special handling for their kickoff - defensive positioning"""
        drawer = self.world.draw
        drawer.annotation((0, 11), "THEIR KICKOFF - DEFENSIVE", drawer.Color.orange, "game_mode")
        
        # Use formation-based positioning
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        # All players maintain defensive positions
        if strategyData.player_unum in [4, 5]:  # Attackers stay at [-2, ±1]
            drawer.annotation(strategyData.mypos, "HOLD DEFENSIVE LINE", drawer.Color.blue, "action_text")
        else:
            drawer.annotation(strategyData.mypos, "DEFENSIVE POSITION", drawer.Color.blue, "action_text")
            
        return self.move(strategyData.my_desired_position,
                        orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))

    def select_skill(self, strategyData):
        drawer = self.world.draw

        # Clear game mode annotation for normal play
        drawer.clear("game_mode")

        # 1. Team Strategy: Dynamic Formation and Role Assignment with caching
        current_time = self.world.time_local_ms
        formation_cache_valid = (current_time - self.last_decision_time < 200 and 
                                self.cached_formation is not None and 
                                self.cached_point_prefs is not None and
                                self.last_play_mode == strategyData.play_mode)
        
        if not formation_cache_valid:
            formation_positions = GenerateDynamicFormation(strategyData)
            point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
            self.cached_formation = formation_positions
            self.cached_point_prefs = point_preferences
            self.last_decision_time = current_time
            self.last_play_mode = strategyData.play_mode
        else:
            formation_positions = self.cached_formation
            point_preferences = self.cached_point_prefs
            
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]

        # FAST orientation calculation
        if strategyData.active_player_unum != strategyData.robot_model.unum:
            target_vec = strategyData.ball_2d - strategyData.my_head_pos_2d
            strategyData.my_desired_orientation = M.vector_angle(target_vec)
        else:
            goal_vec = np.array([15, 0]) - strategyData.my_head_pos_2d
            strategyData.my_desired_orientation = M.vector_angle(goal_vec)

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

        # 2. FAST Decision tree: active or support
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation((0, 10.5), "Active Player: Ball Control", drawer.Color.yellow, "status")

            OPPONENT_GOAL_CENTER = np.array([15, 0])
            mypos = np.array(strategyData.mypos)
            
            # FAST distance calculation using squared distance first
            dx = mypos[0] - OPPONENT_GOAL_CENTER[0]
            dy = mypos[1] - OPPONENT_GOAL_CENTER[1]
            dist_to_goal_sq = dx*dx + dy*dy
            
            ball_pos = np.array(strategyData.ball_2d)
            
            # FAST opponent counting with pre-filter
            num_opponents_nearby = 0
            OPPONENT_THRESHOLD_SQ = 9.0  # 3.0^2
            
            for opp in strategyData.opponent_positions:
                if opp is not None:
                    opp_dx = opp[0] - ball_pos[0]
                    opp_dy = opp[1] - ball_pos[1]
                    if (opp_dx*opp_dx + opp_dy*opp_dy) < OPPONENT_THRESHOLD_SQ:
                        num_opponents_nearby += 1

            # ULTRA-FAST Shooting Decision Tree
            URGENT_SHOOT_DISTANCE_SQ = 64.0  # 8.0^2
            SHOOT_DISTANCE_SQ = 144.0        # 12.0^2
            POSITIONING_DISTANCE_SQ = 196.0  # 14.0^2
            
            # URGENT: Very close to goal - shoot immediately
            if dist_to_goal_sq < URGENT_SHOOT_DISTANCE_SQ:
                # FAST target selection
                if abs(mypos[1]) < 2.0:
                    shoot_target = np.array([15.0, 0])
                else:
                    shoot_target = np.array([15.0, -1.2 if mypos[1] > 0 else 1.2])
                
                drawer.annotation(strategyData.mypos, "URGENT SHOT!", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, shoot_target)

            # NORMAL: Good shooting position
            elif dist_to_goal_sq < SHOOT_DISTANCE_SQ and num_opponents_nearby <= 2:
                # FAST target calculation
                if abs(mypos[1]) < 3.0:
                    shoot_target = np.array([15.0, -0.5 if mypos[1] > 0 else 0.5])
                else:
                    shoot_target = np.array([15.0, -1.0 if mypos[1] > 0 else 1.0])
                
                drawer.annotation(strategyData.mypos, "SHOOT!", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, shoot_target)

            # POSITIONING: Get into better position
            elif dist_to_goal_sq < POSITIONING_DISTANCE_SQ:
                # FAST positioning calculation
                if mypos[1] > 0:
                    pos_target = np.array([min(mypos[0] + 2.0, 13.0), -1.0])
                else:
                    pos_target = np.array([min(mypos[0] + 2.0, 13.0), 1.0])
                
                drawer.annotation(strategyData.mypos, "POSITIONING", drawer.Color.blue, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, pos_target)

            # DEFAULT: Advance toward goal
            else:
                # FAST advance vector calculation
                advance_vec = (OPPONENT_GOAL_CENTER - mypos)
                advance_vec = advance_vec / np.sqrt(dist_to_goal_sq) * 3.0  # Use pre-computed distance
                advance_target = mypos + advance_vec
                
                drawer.annotation(strategyData.mypos, "ADVANCE", drawer.Color.green, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, advance_target)

        else:
            # FAST Support player logic
            drawer.clear("status")
            drawer.clear("action_text")

            teammate_pos = strategyData.teammate_positions[strategyData.player_unum - 1]
            desired_pos = strategyData.my_desired_position

            # FAST distance check using squared distance
            if teammate_pos is None:
                needs_move = True
            else:
                dx = teammate_pos[0] - desired_pos[0]
                dy = teammate_pos[1] - desired_pos[1]
                needs_move = (dx*dx + dy*dy) > 0.25  # 0.5^2

            if needs_move:
                drawer.annotation(strategyData.mypos, "MOVING", drawer.Color.cyan, "action_text")
                return self.move(desired_pos, orientation=strategyData.my_desired_orientation)
            else:
                drawer.annotation(strategyData.mypos, "HOLD POSITION", drawer.Color.blue, "action_text")
                return self.move(desired_pos, orientation=strategyData.my_desired_orientation)
