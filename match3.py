import pygame, draw, random, os

class match3AI:
    def __init__(self,width,height,turns,goal,headless):
        # Game settings
        self.width = width
        self.height = height
        self.BOX_LENGTH = 70

        # Game state
        self.selected = False
        self.selected_x = None
        self.selected_y = None
        self.score = 0
        self.turns_left = turns
        self.GOAL_SCORE = goal
        self.combo_count = 0
        self.exit_game = False
        self.gameover_displayed = False
        self.board = []
        self.headless = headless
        self.anim_dur = 120 #animation duration, bigger number = faster
        self.prediction = ""


        #Centers the opened window on the screen
        os.environ['SDL_VIDEO_CENTERED'] = '1'

        # Board
        self.format_board()
        

        pygame.init()
        if not headless: self.screen = draw.create_screen(self.width, self.height, self.BOX_LENGTH)
        self.clock = pygame.time.Clock()

        if not headless: draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
        if not headless: pygame.display.flip()

    def reset(self,turns,goal):
        # Game state
        self.score = 0
        self.turns_left = turns
        self.GOAL_SCORE = goal
        self.combo_count = 0
        self.exit_game = False
        self.gameover_displayed = False
        self.board = []

        # Board
        self.format_board()
        self.clock = pygame.time.Clock()
        if(not self.headless):draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
        if(not self.headless):pygame.display.flip()

    # -------------------------------------------------------------
    #  BOARD & MATCH FUNCTIONS
    # -------------------------------------------------------------

    def set_prediction(self, prediction):
        self.prediction = prediction

    def match_exists(self):
        # Check rows
        for y in range(len(self.board)):
            streak = 1
            for x in range(1, len(self.board[y])):
                if self.board[y][x] == self.board[y][x-1] and self.board[y][x] != 0:
                    streak += 1
                else:
                    streak = 1
                if streak >= 3:
                    return True

        # Check columns
        for x in range(len(self.board[0])):
            streak = 1
            for y in range(1, len(self.board)):
                if self.board[y][x] == self.board[y-1][x] and self.board[y][x] != 0:
                    streak += 1
                else:
                    streak = 1
                if streak >= 3:
                    return True

        return False

    def format_board(self):
        # self.board = [[random.randrange(1,7) for _ in range(self.width)] for _ in range(self.height)]
        self.board = [[6, 5, 1, 6, 1, 3, 2], [3, 4, 1, 6, 2, 1, 5], [1, 6, 2, 4, 2, 4, 4], [5, 5, 1, 6, 1, 3, 1], [5, 6, 3, 2, 6, 1, 5], [4, 2, 2, 1, 4, 2, 1], [1, 1, 2, 3, 3, 2, 2], [6, 5, 4, 2, 3, 4, 4], [6, 2, 6, 3, 5, 2, 6]]
        while self.match_exists():
            for y in range(len(self.board)):
                for x in range(len(self.board[y])):
                    self.board[y][x] = random.randrange(1,7)

    def show_board(self, grid):
        for row in grid:
            print(" ".join(str(x) for x in row))

    def isAdjacent(self, sx, sy, x, y):
        if sx + 1 == x and sy == y: return True
        if sx - 1 == x and sy == y: return True
        if sy + 1 == y and sx == x: return True
        if sy - 1 == y and sx == x: return True
        return False

    def swap(self,x1, y1, x2, y2):
        self.board[y1][x1], self.board[y2][x2] = self.board[y2][x2], self.board[y1][x1]

    def elim_matches(self):
        elim_list = []

        # Rows
        for y in range(len(self.board)):
            streak = 1
            for x in range(1, len(self.board[y])):
                if self.board[y][x] == self.board[y][x-1] and self.board[y][x] != 0:
                    streak += 1
                else:
                    streak = 1
                if streak == 3:
                    elim_list += [[y,x-2],[y,x-1],[y,x]]
                elif streak > 3:
                    elim_list.append([y,x])

        # Columns
        for x in range(len(self.board[0])):
            streak = 1
            for y in range(1, len(self.board)):
                if self.board[y][x] == self.board[y-1][x] and self.board[y][x] != 0:
                    streak += 1
                else:
                    streak = 1
                if streak == 3:
                    elim_list += [[y-2,x],[y-1,x],[y,x]]
                elif streak > 3:
                    elim_list.append([y,x])

        # Zero out matches
        for y,x in elim_list:
            self.board[y][x] = 0

        return len(elim_list)
    
    def getState(self):
        state = {
            "board":self.board,
            "turns_left":self.turns_left, 
            "score": self.score,
            "goal": self.GOAL_SCORE,
            "combo": self.combo_count,
            "gameover":self.gameover_displayed
        }
        return state

    def board_filled(self):
        for row in self.board:
            if 0 in row:
                return False
        return True

    def drop_tiles(self):
        for x in range(len(self.board[0])):
            for y in range(len(self.board)-2, -1, -1):
                if self.board[y+1][x] == 0 and self.board[y][x] != 0:
                    self.swap(x, y+1, x, y)

    def fill_top(self):
        for x in range(len(self.board[0])):
            if self.board[0][x] == 0:
                self.board[0][x] = random.randrange(1,7)

    # -------------------------------------------------------------
    # GRAPHICS HELPERS
    # -------------------------------------------------------------

    def display_pause(self, x):
        if(not self.headless): pygame.display.flip()
        self.clock.tick(x)

    # -------------------------------------------------------------
    # MAIN GAME LOOP
    # -------------------------------------------------------------

    def playStep(self,agent_move):
        tile1_x, tile1_y,tile2_x,tile2_y = agent_move
        self.combo_count = 1
        self.swap(tile1_x, tile1_y,tile2_x,tile2_y)

        if(not self.headless): draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
        self.display_pause(self.anim_dur)
        
        if self.match_exists():
                self.turns_left -= 1
                turn_score = 0
                self.combo_count = 1

                while self.match_exists():
                    turn_score += self.elim_matches()
                    if(not self.headless):draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                    self.display_pause(self.anim_dur)

                    while not self.board_filled():
                        self.drop_tiles()
                        self.fill_top()
                        if(not self.headless):draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                        self.display_pause(self.anim_dur)

                    self.combo_count += 1

                self.combo_count -= 1
                self.score += turn_score * self.combo_count

                if(not self.headless):draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                if(not self.headless):pygame.display.flip()

                if self.combo_count > 1:
                    if(not self.headless):draw.combo_msg(self.combo_count)
                    if(not self.headless):pygame.display.flip()
                    pygame.time.delay(50)
                    if(not self.headless):draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)

        else:
            # Invalid move → swap back
            self.swap(tile2_x,tile2_y,tile1_x, tile1_y)
            self.turns_left -= 1
            if(not self.headless):draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)

        if self.turns_left <= 0:
            if self.score >= self.GOAL_SCORE:
                if(not self.headless):draw.win()
            else:
                if(not self.headless):draw.lose()

            if(not self.headless):pygame.display.flip()
            self.gameover_displayed = True

    #This is the human played game.
    def run(self):
        while not self.exit_game:
            while self.turns_left > 0 and not self.exit_game:

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.exit_game = True

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = pygame.mouse.get_pos()
                        x = mx // self.BOX_LENGTH
                        y = my // self.BOX_LENGTH

                        if x < self.width and y < self.height:

                            # First tile selected
                            if not self.selected:
                                self.selected = True
                                self.selected_x = x
                                self.selected_y = y
                                draw.selected(x,y)

                            else:
                                # Second tile selected
                                if self.isAdjacent(self.selected_x, self.selected_y, x, y):
                                    self.swap(self.selected_x, self.selected_y, x, y)
                                    draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                                    self.display_pause(2.5)

                                    if self.match_exists():
                                        self.turns_left -= 1
                                        turn_score = 0
                                        self.combo_count = 1

                                        while self.match_exists():
                                            turn_score += self.elim_matches()
                                            draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                                            self.display_pause(2.5)

                                            while not self.board_filled():
                                                self.drop_tiles()
                                                self.fill_top()
                                                draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                                                self.display_pause(2.5)

                                            self.combo_count += 1

                                        self.combo_count -= 1
                                        self.score += turn_score * self.combo_count

                                        draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)
                                        pygame.display.flip()

                                        if self.combo_count > 1:
                                            draw.combo_msg(self.combo_count)
                                            pygame.display.flip()
                                            pygame.time.delay(2000)
                                            draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)

                                    else:
                                        # Invalid move → swap back
                                        self.swap(self.selected_x, self.selected_y, x, y)
                                        draw.window(self.board, self.turns_left, self.score, self.GOAL_SCORE, self.prediction)

                                # Reset selection
                                self.selected = False
                                self.selected_x = None
                                self.selected_y = None

                self.display_pause(60)

            # Game Over screen
            if not self.gameover_displayed:
                if self.score >= self.GOAL_SCORE:
                    draw.win()
                else:
                    draw.lose()

                pygame.display.flip()
                self.gameover_displayed = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_game = True


if __name__ == "__main__":
    width = 7
    height = 9
    turns = 20
    goal = 20
    agent = False #agent = run the game headless or not.

    game = match3AI(width,height,turns,goal,agent)
    game.run()
