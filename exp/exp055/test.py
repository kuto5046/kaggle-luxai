from kaggle_environments import make
import random 
size = random.choice([12, 16, 24, 32])
env = make("lux_ai_2021", configuration={"width": size, "height": size, "annotations": True}, debug=True)
left_agent = 'agent.py'
right_agent = 'random_agent'
steps = env.run([left_agent, right_agent])
# print(steps[0][0]["observation"]["width"], steps[0][0]["observation"]["height"])
# env.render(mode="ipython", width=1000, height=600)