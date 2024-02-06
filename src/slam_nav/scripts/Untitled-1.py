print('장애물 결정')
                    if self.obstacle_type == 'd':
                        # 장애물의 x 좌표가 0보다 큰 경우 ==> 차선을 탈출했다고 판정하고 출발
                        if info.obst_x > 0:
                            self.stop_flag = False

                        # 재정지 해야하는 경우임
                        # 장애물의 x 좌표가 -0.8 인 경우 그리고, 장애물의 좌표가 0.1 보다 큰 경우 거리가 아직 1.3 보다 작은 경우는 정지
                        elif info.obst_x > -0.8 and info.obst_y > 0.1 and dist < 1.3:
                            self.stop_flag = True
                            self.stop()

                    # 정적 장애물인 경우
                    elif info.obst_x < -0.2 or info.obst_x > 0.2:
                        # 안 피해도 되는 장애물임
                        return

                    else:
                        # 임시
                        dy_ob_x = info.obst_y + self.now_pose.x # amcl상에서 장애물의 x좌표
                        dy_ob_y = -info.obst_x + self.now_pose.y # amcl상에서 장애물의 y좌표
                        self.stop_flag  = True
                        self.stop()
                        # 차선 변경 실시
                        
                        # 장애물 옆으로 회피
                        if self.dy_flag == False: # 2차선일 때
                            #1차선으로 이동
                            self.dy_flag = True
                            target_x = dy_ob_x        # amcl상에서 이동해야 할 x좌표
                            target_y = dy_ob_y + 0.35 # amcl상에서 이동해야 할 y좌표
                            self.stop_flag  = False
                            self.create_trajectory( target_x, target_y, dist)
                            

                            #y축을 빼야 하는 경우도 있는걸 염두

                        elif self.dy_flag == True: # 1차선일 때
                            #2차선으로 이동
                            self.dy_flag = False
                            target_x = dy_ob_x        # amcl상에서 이동해야 할 x좌표
                            target_y = dy_ob_y - 0.35 # amcl상에서 이동해야 할 y좌표
                            self.stop_flag  = False
                            self.create_trajectory( target_x, target_y, dist)
                            

                        #y축을 빼야 하는 경우도 있는걸 염두









                        pass