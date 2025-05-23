from turtle import Turtle

ALIGNMENT = 'center'
FONT = ('Courier', 24, 'normal')

class Scoreboard(Turtle):

    def __init__(self, highscore):
        super().__init__()

        self.highscore = highscore
        self.score = 0
        self.color('white')
        self.penup()
        self.goto(0,260)
        self.hideturtle()
        self.update_scoreboard()
        

    def update_scoreboard(self):
        self.write(f'Score: {self.score} Highscore: {self.highscore}', align=ALIGNMENT, font=FONT)

    def game_over(self,highscore):
        self.goto(0,0)
        self.write(f'GAME OVER\nHIGH SCORE: {highscore}', align=ALIGNMENT, font=FONT)

    def increase_score(self):
        self.score += 1
        self.clear()
        self.update_scoreboard()