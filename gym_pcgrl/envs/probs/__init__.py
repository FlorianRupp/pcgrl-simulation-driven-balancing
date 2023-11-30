
from gym_pcgrl.envs.probs.nmmo_prob import NMMO
from gym_pcgrl.envs.probs.nmmo_pcgrl_prob import NMMO_PCGRL
from gym_pcgrl.envs.probs.nmmo_gen_prob import NMMOGen, NMMOGenNPlayers
from gym_pcgrl.envs.probs.nmmo_diff_prob import NMMODiff


# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "nmmo": NMMO,
    "nmmo_gen": NMMOGen,
    "nmmo_gen_nplayers": NMMOGenNPlayers,
    "nmmopcgrl": NMMO_PCGRL,
    "nmmodiff": NMMODiff,
}
