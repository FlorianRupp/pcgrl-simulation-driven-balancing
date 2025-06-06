# Simulation-Driven Balancing of competitive Game Levels with Reinforcement Learning

This is the code base for the journal paper of the same name, published in the IEEE Transaction on Games.

### TL;DR;
* Balance a tile-based game level for 2 players with reinforcement learning by swapping tiles.
* By playing the game multiple times with heuristic agents in simulations, the reinforcement learning
agent learns which tile swaps are most beneficial.
* Example: In this adapted game setting based within the Neural MMO environment, the players must forage
for resources (water/blue and food/dark green) in order to survive longest. Stones/grey and water impede movement. 
By swapping the highlighted tiles, the agent improved the balancing state:

<div style="text-align:center;">
<p align="center">
    <img src="img/example_level.png" alt="Mage Economy" style="width: 54%; display: block; margin-left: auto; margin-right: auto">
</p>
</div>


## Contributions
* An architecture to simulation-driven balance game levels using the PCGRL framework.
* A novel swap-based representation pattern for PCGRL.
  * Implementations can be found in ```gym_pcgrl/envs/reps```
* A study using a derived game setting from the NMMO environment.
  * Implementations for simulation and balancing using PCGRL can be found in ```gym_pcgrl/envs/probs```


### Architecture

The architecture consists of 3 units:
* A level generating unit.
* The balancing unit based on PCGRL.
* A simulator of the game using heuristic agents.

<div style="text-align:center;">
<p align="center">
    <img src="img/Balancing_Architecture.png" alt="Mage Economy" style="width: 84%; margin-right: 10px;">
</p>
</div>


### Demo

The little demo notebook (```balancing_demo.ipynb```) gives a broad overview of the code pipeline used. Trained PPO models for balancing and initial map generation are in ```/models```.



## Limitations
* Computational effort due to the simulation step in each reward calculation.
* Balancing is dependant on heuristic agents:
  * However, we can configure the balancing to balance e.g., for players of different skill or different types of players, mage vs. fighter, for instance.
* Simulating the game can be considered as sampling from the distribution of the true win rate. How often to sample depends
on the environment and heuristics which are used. We use a number of simulations between 10 and 20.

## Update June 2025
* If you want to experiment with the code, we recommend using our reimplementation of the environment, Feast & Forage, along with the improved Markov decision process for balancing, which achieves better overall results and faster convergence during training.
* This code can be found here: [https://github.com/FlorianRupp/feast-and-forage-env/blob/master/README.md](https://github.com/FlorianRupp/feast-and-forage-env/tree/master)


## Further works:
If you are further interested, further work using this code has been published:
* Empirical evaluation with human playtesters: Florian Rupp, Alessandro Puddu, Christian Becker-Asano, and Kai Eckert. It might be balanced, but is it actually good? An Empirical Evaluation of Game Level Balancing. _2024 IEEE Conference on Games (CoG)_, Milan, Italy, pp. 1-4, August 2024. doi: 10.1109/CEC60901.2024.10612054
* Balancing also for asymmetric setups: Florian Rupp and Kai Eckert. Level the Level: Balancing Game Levels for Asymmetric Player Archetypes With Reinforcement Learning. _Proceedings of the 20th International Conference on the Foundations of Digital Games (FDG)_, Graz, Austria, April 2025, doi: 10.1145/3723498.3723747.


## Bibliography

If you use this code, please use this for citations:

```
Journal Paper:
@article{rupp_simulation_2024,
  author={Rupp, Florian and Eberhardinger, Manuel and Eckert, Kai},
  journal={IEEE Transactions on Games}, 
  title={Simulation-Driven Balancing of Competitive Game Levels with Reinforcement Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TG.2024.3399536}}

Conference Paper:
@inproceedings{rupp_balancing_2023,
  author={Rupp, Florian and Eberhardinger, Manuel and Eckert, Kai},
  booktitle={2023 IEEE Conference on Games (CoG)}, 
  title={Balancing of competitive two-player Game Levels with Reinforcement Learning}, 
  year={2023},
  pages={1-8},
  doi={10.1109/CoG57401.2023.10333248}}
```


### Used code

#### Khalifa et al.: Pcgrl: Procedural Content Generation via Reinforcement Learning.

* The code in ```/gym_pcgrl``` is partially taken from the original code base [here](https://github.com/amidos2006/gym-pcgrl) (MIT License).
* For this research it has been extended and adjusted.

```
@inproceedings{khalifa_pcgrl_2020,
	title = {Pcgrl: {Procedural} content generation via reinforcement learning},
	volume = {16},
	booktitle = {Proceedings of the {AAAI} {Conference} on {Artificial} {Intelligence} and {Interactive} {Digital} {Entertainment}},
	author = {Khalifa, Ahmed and Bontrager, Philip and Earle, Sam and Togelius, Julian},
	year = {2020},
	pages = {95--101},
}
```

#### Suarez et al.: The Neural MMO (NMMO) Environment

* The used tiles in ```/gym_pcgrl/envs/probs/nmmo``` is originally from the NMMO environment's official code base [here](https://github.com/NeuralMMO/environment), Version 1.6 (MIT License).
* For this research it has been extended and adjusted.

```
@misc{suarez_neural_2019,
	title = {Neural {MMO}: {A} {Massively} {Multiagent} {Game} {Environment} for {Training} and {Evaluating} {Intelligent} {Agents}},
	author = {Suarez, Joseph and Du, Yilun and Isola, Phillip and Mordatch, Igor},
	year = {2019},
	note = {arXiv:1903.00784}
}
```

#### Suarez: The Neural MMO Environment Baselines

* The code in ```/gym_pcgrl/scripted``` is originally from the NMMO environment's official baselines [here](https://github.com/NeuralMMO/baselines) (MIT License).
* For this research it has been extended and adjusted.
