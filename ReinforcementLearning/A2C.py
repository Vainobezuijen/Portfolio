import numpy as np
import matplotlib.pyplot as plt

L = 20
LEFT_TERMINAL  = -(L+1)
RIGHT_TERMINAL =  (L+1)

rNT = -1
rR  = 20
rL  =  0

gamma = 1.0

def is_terminal(s):
    return (s == LEFT_TERMINAL) or (s == RIGHT_TERMINAL)

def step(s, a):
    s_next = s + a
    if s_next == RIGHT_TERMINAL:
        return s_next, rR
    elif s_next == LEFT_TERMINAL:
        return s_next, rL
    else:
        return s_next, rNT

beta = 0.5

def policy_right_prob(s, theta):
    return 1.0 / (1.0 + np.exp(-beta*(s - theta)))

def sample_action(s, theta):
    p = policy_right_prob(s, theta)
    return +1 if (np.random.rand() < p) else -1

def grad_log_pi(s, a, theta):
    p_r = policy_right_prob(s, theta)
    if a == +1:   # R
        return beta*(1.0 - p_r)
    else:         # L
        return -beta*p_r
    
def state_to_index(s):
    return s + L

v_table = np.zeros(2*L + 1, dtype=np.float64)

def v_func(s):
    return v_table[state_to_index(s)]

def update_v(s, delta):
    v_table[state_to_index(s)] += delta

def generate_trajectories(theta, num_paths=10):
    transitions = []
    for _ in range(num_paths):
        s = np.random.randint(-L, L+1)
        done = False
        while not done:
            a = sample_action(s, theta)
            s_next, r = step(s, a)
            transitions.append((s, a, r, s_next))
            s = s_next
            done = is_terminal(s)
    return transitions

def a2c_update(theta, transitions, alpha_theta=0.01, alpha_v=0.1):
    grad_sum = 0.0
    for (s, a, r, s_next) in transitions:
        if is_terminal(s_next):
            q_hat = r
        else:
            q_hat = r + gamma * v_func(s_next)
        
        A_hat = q_hat - v_func(s)
        
        grad_sum += A_hat * grad_log_pi(s, a, theta)

    grad_avg = grad_sum / len(transitions)
    new_theta = theta + alpha_theta * grad_avg

    for (s, a, r, s_next) in transitions:
        if is_terminal(s_next):
            q_hat = r
        else:
            q_hat = r + gamma * v_func(s_next)
        A_hat = q_hat - v_func(s)
        update_v(s, alpha_v * A_hat)

    return new_theta, grad_avg

num_iterations = 1000
batch_size = 10

theta = 0.0
alpha_theta = 1
alpha_v     = 1

theta_history = []
grad_history  = []

for it in range(num_iterations):
    transitions = generate_trajectories(theta, num_paths=batch_size)

    new_theta, grad_est = a2c_update(
        theta, transitions, alpha_theta=alpha_theta, alpha_v=alpha_v
    )

    theta_history.append(new_theta)
    grad_history.append(grad_est)

    theta = new_theta

iterations = np.arange(1, num_iterations + 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(iterations, theta_history, 'b-o', label='Theta')
plt.xlabel('Iteration')
plt.ylabel('Theta')
plt.title('Theta vs. Iteration')
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(iterations, grad_history, 'r-o', label='dJ/dTheta')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('Gradient')
plt.title('Gradient vs. Iteration')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("Done! Final theta:", theta)
