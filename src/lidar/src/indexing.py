for index, value in enumerate(self.scan_msg.ranges):
            if -45 < degrees[index]< 45 and 0<value<6:
                print(f"obstacle:{degrees[index]}")
                if obstacle_flag ==0 : # 장애물 인덱스 판단을 실시할때
                    obstacle_flag=1
                    start_flag = degrees[index]
                    prev_flag= degrees[index]
                elif obstacle_flag ==1 and abs(degrees[index]-start_flag)<2.0:
                    obstacle_flag=1
                    prev_flag = degrees[index]
                elif obstacle_flag ==1 and abs(degrees[index]-start_flag)>=2.0: #이제는 장애물 한 텀이 끝났다고 생각해야함
                    obstacle_flag=0
                    finish_flag = degrees[index]
                    obstacle_flag=0
                    obstacle_index += 1
                elif abs(degrees[index]-start_flag)>=2.0: 
                    obstacle_index+= 1
