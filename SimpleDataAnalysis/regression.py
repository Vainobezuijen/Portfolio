import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/home/vaino/Documents/Programmeren/Pandas/coaster_db.csv')

df['Speed'] = pd.to_numeric(df['Speed'])
df['Height'] = pd.to_numeric(df['Height'])

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Height
        y = points.iloc[i].Speed
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)

    for i in range(n):
        x = points.iloc[i].Height
        y = points.iloc[i].Speed

        m_gradient += (-2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - m_gradient * L

    return m, b

scaler = StandardScaler
df_stand = scaler.fit_transform(df)

m = 0
b = 0
L = 0.0001
epochs = 100

for i in range(epochs):
    m, b = gradient_descent(m,b,df,L)

print(m, b)

plt.scatter(df.Height, df.Speed, color="black")
plt.plot(list(range(20,80)), [m * x + b for x in range(20,80)], color='red')