import os
import pickle
import pickletools
import numpy
import json 
from itertools import chain

base_dir = "agents/"

with open('data/list_velocity.json') as f:
    d = json.load(f)
    agents_file_vel=list(chain.from_iterable(d.values()))

agents_file = {
    "0":[],
    "4":[],
    "8":[],
    
}   
for subdir in os.listdir(base_dir):
    if ("07-25_16" not in subdir and "07-26" not in subdir and "07-25_17" not in subdir and "07-25_18" not in subdir) or (base_dir+subdir in agents_file_vel) :
        continue  # On passe les dossiers qui ne contiennent pas "07-25"
    
    
    path = os.path.join(base_dir, subdir)
    config_file = os.path.join(path, "config.pkl")
    if os.path.isdir(path) and os.path.isfile(config_file):
        try:
            with open(config_file, "rb") as f:
                config = pickle.load(f)
                print(f"{subdir}: Velocity in state = {config.get('velocity_bool', 'Non défini')}")
                print( f"n looakehad = {config.get('n_lookahead', 'Non défini')}")
                print( f"seed = {config.get('seed', 'Non défini')}")
                agents_file[f"{config.get('n_lookahead', '')}"].append(base_dir+subdir)
        except Exception as e:
            print(f"Erreur dans {subdir} : {e}")
            
file = 'data/list_n_lookahaed_larger_gap.json'

with open(file, "w") as f:
    json.dump(agents_file, f, indent=4)