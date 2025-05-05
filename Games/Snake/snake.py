from turtle import Turtle

STARTING_POSITIONS = [(0,0), (-20,0), (-40,0)]
MOVE_DISTANCE = 20
UP = 90
DOWN = 270
LEFT = 180
RIGHT = 0

class Snake:

    def __init__(self) -> None:
        self.segments = []
        self.create_snake()
        self.head = self.segments[0]
        self.current_heading = RIGHT  # Track the current heading of the snake
        self.can_change_direction = True  # Track if direction change is allowed

    def create_snake(self):
        for position in STARTING_POSITIONS:
            self.add_segment(position)

    def add_segment(self, position):
        new_segment = Turtle('square')
        new_segment.color('white')
        new_segment.penup()
        new_segment.goto(position)
        self.segments.append(new_segment)

    def extend(self):
        self.add_segment(self.segments[-1].position())

    def hide(self):
        for segment in self.segments:
            segment.hideturtle()

    def move(self):
        for seg_num in range(len(self.segments)-1, 0, -1):
            new_x = self.segments[seg_num-1].xcor()
            new_y = self.segments[seg_num-1].ycor()
            self.segments[seg_num].goto(new_x, new_y)
        self.segments[0].forward(MOVE_DISTANCE)
        self.can_change_direction = True  # Allow direction change after moving

    def up(self):
        if self.current_heading != DOWN:
            self.head.setheading(UP)
            self.current_heading = UP
            self.can_change_direction = False  # Lock direction change until next move

    def down(self):
        if self.current_heading != UP:
            self.head.setheading(DOWN)
            self.current_heading = DOWN
            self.can_change_direction = False  # Lock direction change until next move

    def left(self):
        if self.current_heading != RIGHT:
            self.head.setheading(LEFT)
            self.current_heading = LEFT
            self.can_change_direction = False  # Lock direction change until next move

    def right(self):
        if self.current_heading != LEFT:
            self.head.setheading(RIGHT)
            self.current_heading = RIGHT
            self.can_change_direction = False  # Lock direction change until next move