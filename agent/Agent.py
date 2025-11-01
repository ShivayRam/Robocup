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
        
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) 

        self.init_pos = (
            np.array([-12, 0]),     
            np.array([-5, -4]),     
            np.array([-5, 4]),      
            np.array([0, -3]),      
            np.array([0, 3])        
        )[unum-1]
        
        self.last_decision_time = 0
        self.cached_formation = None
        self.cached_point_prefs = None
        self.last_play_mode = None

    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos.copy() 
        self.state = 0

        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) 
        else:
            if self.fat_proxy_cmd is None: 
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: 
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) 


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        
        r = self.world.robot

        if self.fat_proxy_cmd is not None: 
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) 
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) 





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        
        return self.behavior.execute("Dribble",None,None)

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: 
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) 
        else: 
            return self.fat_proxy_kick()


    def kickTarget(self, strategyData, mypos_2d=(0,0),target_2d=(0,0), abort=False, enable_pass_command=False):
        

        vector_to_target = np.array(target_2d) - np.array(mypos_2d)
        
        kick_distance = np.linalg.norm(vector_to_target)
        
        direction_radians = np.arctan2(vector_to_target[1], vector_to_target[0])
        
        kick_direction = np.degrees(direction_radians)


        if strategyData.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: 
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) 
        else: 
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
            self.beam(True) 
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            self.state = 0 if behavior.execute("Get_Up") else 1
        else:
            if strategyData.play_mode == self.world.M_BEFORE_KICKOFF:
                self.beam(avoid_center_circle=True) 
            elif strategyData.play_mode == self.world.M_OUR_KICKOFF:
                self.handle_our_kickoff(strategyData)
            elif strategyData.play_mode == self.world.M_THEIR_KICKOFF:
                self.handle_their_kickoff(strategyData)
            elif strategyData.play_mode == self.world.M_OUR_FREE_KICK:
                self.handle_our_freekick(strategyData)
            elif strategyData.play_mode == self.world.M_THEIR_FREE_KICK:
                self.handle_their_freekick(strategyData)
            elif strategyData.play_mode == self.world.M_OUR_CORNER_KICK:
                self.handle_our_corner(strategyData)
            elif strategyData.play_mode == self.world.M_THEIR_CORNER_KICK:
                self.handle_their_corner(strategyData)
            elif strategyData.play_mode == self.world.M_OUR_GOAL_KICK:
                self.handle_our_goalkick(strategyData)
            elif strategyData.play_mode == self.world.M_THEIR_GOAL_KICK:
                self.handle_their_goalkick(strategyData)
            else:
                self.select_skill(strategyData)


        self.radio.broadcast()

        if self.fat_proxy_cmd is None: 
            self.scom.commit_and_send( strategyData.robot_model.get_command() )
        else: 
            self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
            self.fat_proxy_cmd = ""

    def handle_our_kickoff(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "OUR KICKOFF - ATTACK MODE", drawer.Color.green, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation(strategyData.mypos, "KICKOFF TAKER", drawer.Color.yellow, "action_text")
            return self.kickTarget(strategyData, strategyData.mypos, np.array([10, 0]))
        else:
            return self.move(strategyData.my_desired_position, 
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))

    def handle_their_kickoff(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "THEIR KICKOFF - DEFENSIVE", drawer.Color.orange, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        if strategyData.player_unum in [4, 5]:  
            drawer.annotation(strategyData.mypos, "HOLD DEFENSIVE LINE", drawer.Color.blue, "action_text")
        else:
            drawer.annotation(strategyData.mypos, "DEFENSIVE POSITION", drawer.Color.blue, "action_text")
            
        return self.move(strategyData.my_desired_position,
                        orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))

    def handle_our_freekick(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "OUR FREEKICK - ATTACK SETUP", drawer.Color.green, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        ball_pos = np.array(strategyData.ball_2d)
        
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation(strategyData.mypos, "FREEKICK TAKER", drawer.Color.yellow, "action_text")
            
            OPPONENT_GOAL = np.array([15, 0])
            dist_to_goal = np.linalg.norm(ball_pos - OPPONENT_GOAL)
            
            if ball_pos[0] < -5:  
                best_target = None
                min_dist = float('inf')
                for unum in [4, 5]:
                    teammate_pos = strategyData.teammate_positions[unum-1]
                    if teammate_pos is not None:
                        dist = np.linalg.norm(ball_pos - teammate_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_target = teammate_pos
                
                if best_target is not None:
                    drawer.annotation(strategyData.mypos, "PASS TO ATTACKER", drawer.Color.cyan, "action_text")
                    return self.kickTarget(strategyData, strategyData.mypos, best_target)
                else:
                    drawer.annotation(strategyData.mypos, "CLEAR UP FIELD", drawer.Color.cyan, "action_text")
                    return self.kickTarget(strategyData, strategyData.mypos, OPPONENT_GOAL)
                    
            elif dist_to_goal < 15:  
                if abs(ball_pos[1]) < 3:
                    shoot_target = np.array([15, 0])  
                else:
                    shoot_target = np.array([15, -1.5 if ball_pos[1] > 0 else 1.5])  
                
                drawer.annotation(strategyData.mypos, "DIRECT SHOT!", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, shoot_target)
                
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
                    return self.kickTarget(strategyData, strategyData.mypos, best_target)
                else:
                    advance_target = ball_pos + (OPPONENT_GOAL - ball_pos) * 0.3
                    drawer.annotation(strategyData.mypos, "ADVANCE BALL", drawer.Color.green, "action_text")
                    return self.kickTarget(strategyData, strategyData.mypos, advance_target)
                    
        else:
            if strategyData.player_unum in [4, 5]:  
                current_pos = np.array(strategyData.mypos)
                desired_pos = strategyData.my_desired_position
                
                if np.linalg.norm(current_pos - desired_pos) < 1.0:
                    run_target = desired_pos + np.array([3, 0])  
                    run_target[0] = min(run_target[0], 13)  
                    drawer.annotation(strategyData.mypos, "MAKING RUN", drawer.Color.magenta, "action_text")
                    return self.move(run_target, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))
            
            drawer.annotation(strategyData.mypos, "HOLD POSITION", drawer.Color.blue, "action_text")
            return self.move(strategyData.my_desired_position,
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d))

    def handle_their_freekick(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "THEIR FREEKICK - DEFENSIVE SETUP", drawer.Color.orange, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        ball_pos = np.array(strategyData.ball_2d)
        
        if strategyData.player_unum in [2, 3]:  
            drawer.annotation(strategyData.mypos, "WALL DEFENDER", drawer.Color.red, "action_text")
            
            desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos)
            return self.move(strategyData.my_desired_position, orientation=desired_orientation)
            
        elif strategyData.player_unum == 1:  
            drawer.annotation(strategyData.mypos, "GK - READY", drawer.Color.red, "action_text")
            
            ball_to_goal_vec = np.array([-14, 0]) - ball_pos
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
                drawer.annotation(strategyData.mypos, "MARKING OPPONENT", drawer.Color.orange, "action_text")
                mark_pos = closest_opponent - (closest_opponent - ball_pos) * 0.3
                return self.move(mark_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            else:
                drawer.annotation(strategyData.mypos, "COVERING SPACE", drawer.Color.blue, "action_text")
                return self.move(strategyData.my_desired_position,
                               orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))

    def handle_our_corner(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "OUR CORNER - ATTACKING", drawer.Color.green, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        ball_pos = np.array(strategyData.ball_2d)
        ball_y = ball_pos[1]
        
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation(strategyData.mypos, "CORNER TAKER", drawer.Color.yellow, "action_text")
            
            if ball_y > 0:  
                cross_target = np.array([-8, -3])
                drawer.annotation(strategyData.mypos, "CROSS TO FAR POST", drawer.Color.cyan, "action_text")
            else:  
                cross_target = np.array([-8, 3])
                drawer.annotation(strategyData.mypos, "CROSS TO FAR POST", drawer.Color.cyan, "action_text")
            
            return self.kickTarget(strategyData, strategyData.mypos, cross_target)
            
        else:
            if strategyData.player_unum == 1:  
                drawer.annotation(strategyData.mypos, "GK - STAY READY", drawer.Color.blue, "action_text")
            elif strategyData.player_unum in [2, 3]:  
                current_pos = np.array(strategyData.mypos)
                if np.linalg.norm(current_pos - strategyData.my_desired_position) < 1.0:
                    run_target = strategyData.my_desired_position + np.array([2, 0])
                    drawer.annotation(strategyData.mypos, "MAKING RUN", drawer.Color.magenta, "action_text")
                    return self.move(run_target, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
                else:
                    drawer.annotation(strategyData.mypos, "HOLD POSITION", drawer.Color.blue, "action_text")
            else:  
                drawer.annotation(strategyData.mypos, "ATTACKING POSITION", drawer.Color.red, "action_text")
            
            return self.move(strategyData.my_desired_position,
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))

    def handle_their_corner(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "THEIR CORNER - DEFENDING", drawer.Color.orange, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        ball_pos = np.array(strategyData.ball_2d)
        
        if strategyData.player_unum == 1:  
            drawer.annotation(strategyData.mypos, "GK - CORNER DEFENSE", drawer.Color.red, "action_text")
            
            if ball_pos[1] > 0:  
                gk_pos = np.array([-14, 1.0])  
            else:  
                gk_pos = np.array([-14, -1.0])  
                
            return self.move(gk_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            
        elif strategyData.player_unum in [2, 3]:  
            drawer.annotation(strategyData.mypos, "ZONAL MARKING", drawer.Color.orange, "action_text")
            return self.move(strategyData.my_desired_position, 
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            
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
                drawer.annotation(strategyData.mypos, "MAN MARKING", drawer.Color.red, "action_text")
                mark_pos = closest_opponent + (closest_opponent - ball_pos) * 0.1
                return self.move(mark_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            else:
                drawer.annotation(strategyData.mypos, "COVER SPACE", drawer.Color.blue, "action_text")
                return self.move(strategyData.my_desired_position,
                               orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))

    def handle_our_goalkick(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "OUR GOAL KICK - BUILD UP", drawer.Color.green, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        ball_pos = np.array(strategyData.ball_2d)
        
        if strategyData.player_unum == 1:
            drawer.annotation(strategyData.mypos, "GOAL KICK TAKER", drawer.Color.yellow, "action_text")
            
            best_target = None
            min_opponent_pressure = float('inf')
            
            for i, teammate_pos in enumerate(strategyData.teammate_positions):
                if teammate_pos is not None and i+1 != 1:  
                    opponent_pressure = 0
                    for opp_pos in strategyData.opponent_positions:
                        if opp_pos is not None:
                            dist_to_teammate = np.linalg.norm(opp_pos - teammate_pos)
                            if dist_to_teammate < 3.0:
                                opponent_pressure += (3.0 - dist_to_teammate)
                    
                    if opponent_pressure < min_opponent_pressure:
                        min_opponent_pressure = opponent_pressure
                        best_target = teammate_pos
            
            if best_target is not None and min_opponent_pressure < 2.0:
                drawer.annotation(strategyData.mypos, "PASS TO TEAMMATE", drawer.Color.cyan, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, best_target)
            else:
                clear_target = np.array([0, 0])
                drawer.annotation(strategyData.mypos, "CLEAR LONG", drawer.Color.cyan, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, clear_target)
                
        else:
            drawer.annotation(strategyData.mypos, "PROVIDING OPTION", drawer.Color.blue, "action_text")
            return self.move(strategyData.my_desired_position,
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))

    def handle_their_goalkick(self, strategyData):
        drawer = self.world.draw
        drawer.annotation((0, 11), "THEIR GOAL KICK - PRESSING", drawer.Color.orange, "game_mode")
        
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]
        
        ball_pos = np.array(strategyData.ball_2d)
        
        if strategyData.player_unum in [4, 5]:  
            drawer.annotation(strategyData.mypos, "HIGH PRESS", drawer.Color.red, "action_text")
            
            closest_opponent = None
            min_dist = 6.0
            
            for i, opp_pos in enumerate(strategyData.opponent_positions):
                if opp_pos is not None and opp_pos[0] > -10:  
                    dist = np.linalg.norm(np.array(strategyData.mypos) - opp_pos)
                    if dist < min_dist:
                        min_dist = dist
                        closest_opponent = opp_pos
            
            if closest_opponent is not None and min_dist < 5.0:
                press_pos = closest_opponent + (closest_opponent - ball_pos) * 0.2
                return self.move(press_pos, orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
            else:
                return self.move(strategyData.my_desired_position,
                               orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))
                               
        else:  
            drawer.annotation(strategyData.mypos, "ZONAL PRESS", drawer.Color.orange, "action_text")
            return self.move(strategyData.my_desired_position,
                           orientation=strategyData.GetDirectionRelativeToMyPositionAndTarget(ball_pos))

    def select_skill(self, strategyData):
        drawer = self.world.draw

        drawer.clear("game_mode")

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

        if strategyData.active_player_unum != strategyData.robot_model.unum:
            target_vec = strategyData.ball_2d - strategyData.my_head_pos_2d
            strategyData.my_desired_orientation = M.vector_angle(target_vec)
        else:
            goal_vec = np.array([15, 0]) - strategyData.my_head_pos_2d
            strategyData.my_desired_orientation = M.vector_angle(goal_vec)

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation((0, 10.5), "Active Player: Ball Control", drawer.Color.yellow, "status")

            OPPONENT_GOAL_CENTER = np.array([15, 0])
            mypos = np.array(strategyData.mypos)
            
            dx = mypos[0] - OPPONENT_GOAL_CENTER[0]
            dy = mypos[1] - OPPONENT_GOAL_CENTER[1]
            dist_to_goal_sq = dx*dx + dy*dy
            
            ball_pos = np.array(strategyData.ball_2d)
            
            num_opponents_nearby = 0
            OPPONENT_THRESHOLD_SQ = 9.0  
            
            for opp in strategyData.opponent_positions:
                if opp is not None:
                    opp_dx = opp[0] - ball_pos[0]
                    opp_dy = opp[1] - ball_pos[1]
                    if (opp_dx*opp_dx + opp_dy*opp_dy) < OPPONENT_THRESHOLD_SQ:
                        num_opponents_nearby += 1

            URGENT_SHOOT_DISTANCE_SQ = 64.0  
            SHOOT_DISTANCE_SQ = 144.0        
            POSITIONING_DISTANCE_SQ = 196.0  
            
            if dist_to_goal_sq < URGENT_SHOOT_DISTANCE_SQ:
                if abs(mypos[1]) < 2.0:
                    shoot_target = np.array([15.0, 0])
                else:
                    shoot_target = np.array([15.0, -1.2 if mypos[1] > 0 else 1.2])
                
                drawer.annotation(strategyData.mypos, "URGENT SHOT!", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, shoot_target)

            elif dist_to_goal_sq < SHOOT_DISTANCE_SQ and num_opponents_nearby <= 2:
                if abs(mypos[1]) < 3.0:
                    shoot_target = np.array([15.0, -0.5 if mypos[1] > 0 else 0.5])
                else:
                    shoot_target = np.array([15.0, -1.0 if mypos[1] > 0 else 1.0])
                
                drawer.annotation(strategyData.mypos, "SHOOT!", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, shoot_target)

            elif dist_to_goal_sq < POSITIONING_DISTANCE_SQ:
                if mypos[1] > 0:
                    pos_target = np.array([min(mypos[0] + 2.0, 13.0), -1.0])
                else:
                    pos_target = np.array([min(mypos[0] + 2.0, 13.0), 1.0])
                
                drawer.annotation(strategyData.mypos, "POSITIONING", drawer.Color.blue, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, pos_target)

            else:
                advance_vec = (OPPONENT_GOAL_CENTER - mypos)
                advance_vec = advance_vec / np.sqrt(dist_to_goal_sq) * 3.0  
                advance_target = mypos + advance_vec
                
                drawer.annotation(strategyData.mypos, "ADVANCE", drawer.Color.green, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, advance_target)

        else:
            drawer.clear("status")
            drawer.clear("action_text")

            teammate_pos = strategyData.teammate_positions[strategyData.player_unum - 1]
            desired_pos = strategyData.my_desired_position

            if teammate_pos is None:
                needs_move = True
            else:
                dx = teammate_pos[0] - desired_pos[0]
                dy = teammate_pos[1] - desired_pos[1]
                needs_move = (dx*dx + dy*dy) > 0.25  

            if needs_move:
                drawer.annotation(strategyData.mypos, "MOVING", drawer.Color.cyan, "action_text")
                return self.move(desired_pos, orientation=strategyData.my_desired_orientation)
            else:
                drawer.annotation(strategyData.mypos, "HOLD POSITION", drawer.Color.blue, "action_text")
                return self.move(desired_pos, orientation=strategyData.my_desired_orientation)