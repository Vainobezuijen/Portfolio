from turtle import Screen
from snake import Snake
from food import Food
from scoreboard import Scoreboard
from score import Score

import time

def update_highscore():
    with open('highscore.txt', 'r') as file:
        current_highscore = file.readlines()
    with open('highscore.txt', 'w') as file:
        highscore = int(current_highscore[0])
        if scoreboard.score > int(current_highscore[0]):
            highscore = scoreboard.score
            file.write(str(highscore))
        else:
            file.write(str(highscore))
    return highscore

screen = Screen()
screen.setup(width = 600, height=600)
screen.bgcolor('black')
screen.title('Snake')
screen.tracer(0)

with open('highscore.txt', 'r') as file:
    current_highscore = file.readlines()
    current_highscore = int(current_highscore[0])

snake = Snake()
food = Food()
scoreboard = Scoreboard(current_highscore)

screen.listen()
screen.onkey(snake.up, 'Up')
screen.onkey(snake.down, 'Down')
screen.onkey(snake.left, 'Left')
screen.onkey(snake.right, 'Right')

screen.update()

game_is_on = True
last_move_time = time.time()  # Track time of last snake movement
move_interval = 0.3

while game_is_on:
    screen.update()
    current_time = time.time()
    if current_time - last_move_time > move_interval:
        last_move_time = current_time
        snake.move()

        # Eating food
        if snake.head.distance(food) < 15:
            food.refresh()
            snake.extend()
            scoreboard.increase_score()
            move_interval *= 0.98

        # Collision wall
        if snake.head.xcor() > 280 or snake.head.xcor() < -280 or snake.head.ycor() > 280 or snake.head.ycor() <-280:
            highscore = update_highscore()
            snake.hide()
            scoreboard.game_over(highscore)
            game_is_on = False

        # Collision tail
        for segment in snake.segments[1:]:
            if snake.head.distance(segment) < 5:
                highscore = update_highscore()
                snake.hide()
                scoreboard.game_over(highscore)
                game_is_on = False

screen.exitonclick()