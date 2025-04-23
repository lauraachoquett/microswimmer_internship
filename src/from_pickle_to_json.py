import json
import os
import pickle


def convertir_pickle_en_json_recursif(repertoire):
    """
    Convertit tous les fichiers pickle dans un répertoire donné (et ses sous-dossiers) en fichiers JSON.

    Args:
        repertoire (str): Chemin du répertoire contenant les fichiers pickle.
    """
    print("Conversion des fichiers pickle en JSON (récursif)...")
    # Parcourt tous les fichiers et sous-dossiers du répertoire
    for racine, sous_dossiers, fichiers in os.walk(repertoire):
        for fichier in fichiers:
            chemin_fichier = os.path.join(racine, fichier)
            # Vérifie si le fichier est un fichier pickle
            if fichier.endswith(".pkl") or fichier.endswith(".pickle"):
                try:
                    # Chargement des données depuis le fichier pickle
                    with open(chemin_fichier, "rb") as fichier_pickle:
                        donnees = pickle.load(fichier_pickle)

                    # Chemin pour le fichier JSON de sortie
                    fichier_json = fichier.replace(".pkl", ".json").replace(
                        ".pickle", ".json"
                    )
                    chemin_fichier_json = os.path.join(racine, fichier_json)

                    # Sauvegarde des données au format JSON
                    with open(chemin_fichier_json, "w") as fichier_json:
                        json.dump(donnees, fichier_json, indent=4)

                    print(f"Converti : {chemin_fichier} -> {chemin_fichier_json}")
                except Exception as e:
                    print(f"Erreur lors de la conversion de {chemin_fichier} : {e}")


# Exemple d'utilisation
repertoire_pickle = "agents/"  # Remplacez par le chemin de votre répertoire
convertir_pickle_en_json_recursif(repertoire_pickle)
