import os
import pickle


import os
import pickle

base_dir = "agents/"

for subdir in os.listdir(base_dir):
    path = os.path.join(base_dir, subdir)
    config_file = os.path.join(path, "config.pkl")

    if os.path.isdir(path) and os.path.isfile(config_file):
        try:
            with open(config_file, "rb") as f:
                config = pickle.load(f)
            print(f"{subdir}: Velocity in state = {config.get('velocity_bool', 'Non défini')}")
            print( f"n looakehad = {config.get('n_lookahead', 'Non défini')}")
        except Exception as e:
            print(f"Erreur dans {subdir} : {e}")