import matplotlib.pyplot as plt
import numpy as np

def hist_scientific(data, xlabel='', ylabel='Fréquence', title='', 
                   bins='auto', save_name=None):
    """
    Crée un histogramme propre pour publication scientifique
    
    Parameters:
    -----------
    data : array-like
        Données à représenter
    xlabel, ylabel, title : str
        Labels et titre
    bins : int ou str
        Nombre de bins ou 'auto'
    save_name : str ou None
        Nom du fichier de sauvegarde (sans extension)
    """
    
    # Configuration basique
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'font.family': 'serif',
    })
    
    # Figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Histogramme
    ax.hist(data, bins=bins, 
            alpha=0.7, 
            color='steelblue', 
            edgecolor='black', 
            linewidth=0.8)
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Grille et mise en page
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarde
    if save_name:
        plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Stats de base
    print(f"N = {len(data)}, μ = {np.mean(data):.2f}, σ = {np.std(data):.2f}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Données d'exemple
    data = np.random.normal(100, 15, 1000)
    
    # Appel simple
    hist_scientific(data, 
                   xlabel='Mesure (unité)', 
                   ylabel='Fréquence',
                   title='Distribution expérimentale',
                   save_name='mon_histogramme')