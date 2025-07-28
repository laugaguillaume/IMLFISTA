# Turoriel : code multiniveaux avec débruiteur conditionnel

## 1- Télécharger le code depuis Git
Le code est sur le repository Github IMLFISTA créé par Guillaume et Nils, disponible *via* le lien suivant : https://github.com/laugaguillaume/IMLFISTA.

Pour télécharger le code, dans le terminal :
1. Se placer dans le dossier où vous voulez télécharger le repository
2. Télécharger le repository avec la commande
    ```bash
    git clone https://github.com/laugaguillaume/IMLFISTA.git
    ```
3. Se placer dans le dossier créé par Git avec
    ```bash
    cd IMLFISTA
    ```

Vous avez maintenant accès au code de la branche ```main```, qui contient le code de Guillaume pour IMLFISTA et celui de Nils pour le Multilevel Plug and Play. 

Pour changer de branche et accéder à notre code qui se trouve dans la branche ```edgar```, exécuter dans le terminal :
```bash
git checkout edgar
```

## 2- Exécuter le code

### Installer les librairies nécessaires

Pour installer les dépendances dans votre installation Python locale, exécuter :
```bash
pip install -r requirements.txt
```

Si vous préférez utiliser un environnement virtuel pour ne pas modifier votre installation locale, vous pouvez en créer un et installer les dépendances dedans, en exécutant :
```bash
python3 -m venv imlfista        # Crée l'environnement virtuel "imlfista"
source imlfista/bin/activate    # Active l'environnement virtuel
pip install -r requirements.txt # Installe les dépendances dans l'environnement virtuel
```

### Exécuter le code

Il faut exécuter le code comme un module avec ```python3 -m``` car les scripts du dossier ```demo``` dépendent des fichiers du dossier ```mutltilevel```. Par exemple pour exécuter notre code :
```bash
python3 -m demo.demo_multilevel_conditional_denoising
```