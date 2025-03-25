from turtle import Turtle
import random

class Score(Turtle):

    def __init__(self, score):
        super().__init__()
        self.penup()
        self.goto(50,260)
        self.hideturtle()
        self.score = score

    def show_score(self):
        self.write(f'HIGH SCORE: {self.score}')