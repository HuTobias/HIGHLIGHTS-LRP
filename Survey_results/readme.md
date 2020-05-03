# Evaluation of the survey results

This folder contains the evaluation of the survey results.

The raw results, as exported from lime-survey, are saved in *results_all_grps.csv*, *results_grp3.csv* and *results_grp4.csv*.
The module *result_fusion.py* combines all those files to one (*raw_fusion.csv*), removes uneccessary columns,checks whether the participants chose the right agents during the trust task and calculates the score that they achieved during the object selection part of the retrospection task (*fusion_final.csv*).

Except for the textual information, all main values are evaluated in *evaluation.py*.
This module also defines some functions used by the other evaluation scripts.
Most importantly, the functions needed for the Mann Whitney tests and the *show_and_save_plt* function that defines how all plots look like.

*evalutate_demographic.py* analyzes the demographic questions about age,gender, Pacman experience and AI experience.

The concepts found in the participants justifications for their decisions and textual descriptions of the agent's strategy are stored in the *PacmanStrategies_INT* and *PacmanStrategies_TRUST* files.
This data is evaluated in *evaluate_text.py*, which also defines how those concepts are grouped together and how the participants score is defined based on those concepts.
The *text_intention* and *text_trust* show the results of this grouping and the score per agent.

All resulting graphs are saved in the folder *figures*.
