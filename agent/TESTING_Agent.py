from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np

from strategy.Assignment import role_assignment 
from strategy.Strategy import Strategy 
from formation.Formation import GenerateBasicFormation, GenerateDynamicFormation


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type for 5v5
        # 0 = GK, 1 = DEF, 2 = MID, 3 = FWD
        robot_type = (0, 1, 1, 2, 3)[unum-1]

        # Initialize base agent
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        # NEW: Updated initial formation using the basic formation
        formation = GenerateBasicFormation()
        self.init_pos = formation[unum]  # Get position for this player number
        
        # Store formation types for different situations
        self.current_formation = "basic"
        self.formation_history = []

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

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target)

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

    def kickTarget(self, strategyData, mypos_2d=(0,0), target_2d=(0,0), abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick towards target
        '''
        # Calculate the vector from the current position to the target position
        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        # Calculate the distance (magnitude of the vector)
        kick_distance = np.linalg.norm(vector_to_target)
        
        # Calculate the direction (angle) in radians
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        # Convert direction to degrees for easier interpretation
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
            if strategyData.play_mode != self.world.M_BEFORE_KICKOFF:
                self.select_skill(strategyData)
            else:
                # During before kickoff, use basic formation
                formation = GenerateBasicFormation()
                target_pos = formation[self.world.robot.unum]
                self.move(target_pos, orientation=0)

        # Broadcast and send to server
        self.radio.broadcast()
        
        if self.fat_proxy_cmd is None: # normal behavior
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: # fat proxy behavior
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""

    def select_skill(self, strategyData):
        # Decision making with enhanced formations
        drawer = self.world.draw
        path_draw_options = self.path_manager.draw_options

        # Import enhanced formation functions
        from formation.Formation import GenerateDynamicFormation, GetPlayerRole
        
        # Determine game state for dynamic formation
        ball_pos = strategyData.ball_abs_pos[:2]
        is_offensive = ball_pos[0] > -5  # Ball in opponent half
        
        # Generate appropriate formation
        formation_positions = GenerateDynamicFormation(
            ball_pos, 
            is_offensive,
            strategyData.play_mode
        )
        
        # Store current formation type for debugging
        if is_offensive:
            self.current_formation = "offensive"
        else:
            self.current_formation = "defensive"
            
        # Draw formation information
        role = GetPlayerRole(strategyData.player_unum, self.current_formation)
        drawer.annotation((0, 10), f"Role: {role}", drawer.Color.cyan, "role_info")
        drawer.annotation((0, 9.5), f"Formation: {self.current_formation}", drawer.Color.cyan, "formation_info")

        # Role Assignment
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation((0, 10.5), "Role Assignment Phase", drawer.Color.yellow, "status")
        else:
            drawer.clear("status")

        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        strategyData.my_desried_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.my_desired_position)

        # Draw formation positions and connections
        for player_num, position in formation_positions.items():
            color = drawer.Color.blue if player_num == strategyData.player_unum else drawer.Color.green
            drawer.circle(position, 0.3, 2, color, f"formation_pos_{player_num}")
            drawer.annotation((position[0], position[1] + 0.5), f"P{player_num}", color, f"formation_label_{player_num}")

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target_line")

        if not strategyData.IsFormationReady(point_preferences):
            return self.move(strategyData.my_desired_position, orientation=strategyData.my_desried_orientation)

        # Decision making based on game state and role
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            # Active player decision making
            drawer.annotation((0, 10.5), "Active Player - Decision Making", drawer.Color.yellow, "status")
            
            # Enhanced passing strategy
            pass_reciever_unum = self._select_best_pass_target(strategyData)
            if pass_reciever_unum is not None:
                target = strategyData.teammate_positions[pass_reciever_unum-1]
                drawer.line(strategyData.mypos, target, 2, drawer.Color.red, "pass_line")
                drawer.annotation(target, f"Pass to P{pass_reciever_unum}", drawer.Color.red, "pass_target")
                return self.kickTarget(strategyData, strategyData.mypos, target)
            else:
                # No good pass, shoot at goal
                target = (15, 0)
                drawer.line(strategyData.mypos, target, 3, drawer.Color.orange, "shot_line")
                drawer.annotation((0, 9), "SHOOTING!", drawer.Color.orange, "shot_text")
                return self.kickTarget(strategyData, strategyData.mypos, target)
        else:
            # Supporting player behavior
            drawer.clear("pass_line")
            drawer.clear("shot_line")
            drawer.clear("shot_text")
            
            # Support players move to formation and face the ball
            return self.move(strategyData.my_desired_position, orientation=strategyData.ball_dir)

    def _select_best_pass_target(self, strategyData):
        """
        Enhanced pass target selection considering:
        - Teammate positions
        - Opponent positions
        - Field position
        - Game situation
        """
        best_target = None
        best_score = -1
        
        for i in range(1, 6):  # Check all teammates
            if i == strategyData.player_unum:
                continue
                
            if i-1 < len(strategyData.teammate_positions) and strategyData.teammate_positions[i-1] is not None:
                teammate_pos = strategyData.teammate_positions[i-1]
                
                # Calculate pass score
                score = self._calculate_pass_score(strategyData, teammate_pos, i)
                
                if score > best_score:
                    best_score = score
                    best_target = i
                    
        # Only pass if score is above threshold
        if best_score > 0.5:
            return best_target
        else:
            return None  # Indicates no good pass available

    def _calculate_pass_score(self, strategyData, target_pos, target_unum):
        """
        Calculate how good a pass to this target would be
        Higher score = better pass
        """
        score = 0.0
        
        # Base score based on target's field position (prefer forward passes)
        if target_pos[0] > strategyData.mypos[0]:  # Forward pass
            score += 0.3
        else:  # Backward pass
            score -= 0.2
            
        # Distance factor (prefer medium distance passes)
        distance = np.linalg.norm(np.array(strategyData.mypos) - np.array(target_pos))
        if 3.0 <= distance <= 8.0:
            score += 0.3
        elif distance < 3.0:
            score += 0.1
        else:  # Too far
            score -= 0.2
            
        # Check if target is open (no opponents nearby)
        min_opponent_dist = float('inf')
        for opp_pos in strategyData.opponent_positions:
            if opp_pos is not None:
                dist = np.linalg.norm(np.array(target_pos) - np.array(opp_pos))
                min_opponent_dist = min(min_opponent_dist, dist)
                
        if min_opponent_dist > 3.0:
            score += 0.4  # Very open
        elif min_opponent_dist > 2.0:
            score += 0.2  # Somewhat open
        else:
            score -= 0.3  # Too crowded
            
        return max(0.0, score)  # Ensure non-negative score

    # Fat proxy auxiliary methods (keep existing)
    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3)
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True)
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")