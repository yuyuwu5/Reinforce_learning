### How to train pg
* python3.7 main.py --train\_pg

### How to train DQN
* python3.7 main.py --train\_dqn

### How to plot reward
* import matplotlib.pyplot in train\_pg.py and train\_dqn.py
#### For report Q1
* Unmark plt.plot in code then can plot reward
#### For report Q2
* Set different target network update frequency manually
* Save reward vs episode in pickle
* Set file name manualy in plot\_target.py then it can get desired picture
#### For report Q3
* Train on different network and save reward. Then plot picture
* python3.7 main.py --train\_ddqn
* python3.7 main.py --train\_duel
### For bonus
Put all data need and glove vector in data directory
#### Training
* python3.7 train.py to train RL model
#### Predicting
* python3.7 S2S\_attention\_prediction.py --test\_data\_path path --output\_path output
