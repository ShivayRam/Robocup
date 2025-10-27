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

        #self.init_pos = GenerateDynamicFormation(Strategy(self.world))[unum-1]
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
        formation_positions = GenerateDynamicFormation(strategyData)
        point_preferences = role_assignment(strategyData.teammate_positions, formation_positions)
        strategyData.my_desired_position = point_preferences[strategyData.player_unum]

        if strategyData.active_player_unum != strategyData.robot_model.unum:
            strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(strategyData.ball_2d)
        else:
            strategyData.my_desired_orientation = strategyData.GetDirectionRelativeToMyPositionAndTarget(np.array([15, 0]))

        drawer.line(strategyData.mypos, strategyData.my_desired_position, 2, drawer.Color.blue, "target line")

        # 2. Decision tree: active or support
        if strategyData.active_player_unum == strategyData.robot_model.unum:
            drawer.annotation((0, 10.5), "Active Player: Ball Control", drawer.Color.yellow, "status")

            OPPONENT_GOAL_CENTER = np.array([15, 0])
            mypos = np.array(strategyData.mypos)
            dist_to_goal = np.linalg.norm(mypos - OPPONENT_GOAL_CENTER)

            ball_pos = np.array(strategyData.ball_2d)
            ball_vel = np.array(getattr(strategyData, 'ball_vel_2d', [0.0, 0.0]))
            # Count nearby defenders/opponents around ball
            num_opponents_nearby = sum(
                1 for opp in strategyData.opponent_positions if opp is not None
                and np.linalg.norm(np.array(opp) - ball_pos) < 3.0
            )

            SHOOT_DISTANCE_THRESHOLD = 7.0
            if dist_to_goal < SHOOT_DISTANCE_THRESHOLD and num_opponents_nearby <= 1:
                # Choose shoot directly
                if mypos[1] > 0:
                    shoot_target = np.array([15.0, -0.8])
                else:
                    shoot_target = np.array([15.0,  0.8])

                drawer.annotation(strategyData.mypos, "FAST SHOT!", drawer.Color.red, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, shoot_target)

            ATTACK_ZONE_DISTANCE = 12.0
            if dist_to_goal < ATTACK_ZONE_DISTANCE:
                dribble_vec = (OPPONENT_GOAL_CENTER - mypos)
                dribble_vec = dribble_vec / np.linalg.norm(dribble_vec) * 3.0
                dribble_target = mypos + dribble_vec

                drawer.annotation(strategyData.mypos, "QUICK DRIBBLE", drawer.Color.green, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, dribble_target)

            BALL_PREDICT_DT = 0.5
            predicted_ball = ball_pos + ball_vel * BALL_PREDICT_DT

            if dist_to_goal < 14.0 and num_opponents_nearby == 0:
                if mypos[1] > 0:
                    align_target = np.array([15.0, -0.5])
                else:
                    align_target = np.array([15.0,  0.5])

                drawer.annotation(mypos, "ALIGN SHOT PATH", drawer.Color.blue, "action_text")
                return self.kickTarget(strategyData, strategyData.mypos, align_target)

            # Default: dribble toward goal
            dribble_vec = (OPPONENT_GOAL_CENTER - mypos)
            dribble_vec = dribble_vec / np.linalg.norm(dribble_vec) * 2.0
            dribble_target = mypos + dribble_vec

            drawer.annotation(strategyData.mypos, "DRIBBLE", drawer.Color.green, "action_text")
            return self.kickTarget(strategyData, strategyData.mypos, dribble_target)

        else:
            # Support player logic
            drawer.clear("status")
            drawer.clear("action_text")

            teammate_pos = strategyData.teammate_positions[strategyData.player_unum - 1]
            desired_pos = strategyData.my_desired_position

            if teammate_pos is None or np.linalg.norm(teammate_pos - desired_pos) > 0.5:
                drawer.annotation(strategyData.mypos, "MOVING", drawer.Color.cyan, "action_text")
                return self.move(desired_pos, orientation=strategyData.my_desired_orientation)
            else:
                drawer.annotation(strategyData.mypos, "HOLD", drawer.Color.blue, "action_text")
                return self.move(desired_pos, orientation=strategyData.my_desired_orientation)