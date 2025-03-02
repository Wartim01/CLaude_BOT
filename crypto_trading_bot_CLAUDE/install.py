# install.py
"""
Script d'installation et de configuration du bot de trading
"""
import os
import sys
import subprocess
import argparse
import json
import getpass

def check_python_version():
    """Vérifie la version de Python"""
    if sys.version_info < (3, 8):
        print("Erreur: Python 3.8 ou supérieur est requis")
        sys.exit(1)
    print(f"Python {sys.version} détecté")

def install_dependencies():
    """Installe les dépendances requises"""
    print("Installation des dépendances...")
    requirements = [
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "requests>=2.25.0",
        "python-dotenv>=0.19.0",
        "websocket-client>=1.2.0",
        "argparse>=1.4.0"
    ]
    
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dépendances installées avec succès")
    except subprocess.CalledProcessError:
        print("Erreur lors de l'installation des dépendances")
        sys.exit(1)

def create_directories():
    """Crée les répertoires nécessaires"""
    directories = [
        "data",
        "data/market_data",
        "data/trade_logs",
        "data/performance",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Répertoires créés")

def setup_configuration():
    """Configure les paramètres du bot"""
    print("\n=== Configuration du bot de trading ===\n")
    
    use_testnet = input("Utiliser le réseau de test Binance? (O/n): ").lower() != "n"
    
    if use_testnet:
        print("\nVous allez utiliser le réseau de test Binance.")
        print("Rendez-vous sur https://testnet.binance.vision/ pour créer des clés API de test.")
    else:
        print("\nATTENTION: Vous allez utiliser le réseau de production Binance.")
        print("Le bot pourra trader avec de vrais fonds!")
    
    # Remplacer getpass.getpass par input standard
    print("\nNOTE: Vous allez entrer des informations sensibles. Assurez-vous que personne ne regarde votre écran.")
    api_key = input("Clé API Binance: ")
    api_secret = input("Clé secrète API Binance: ")
    
    # Créer le fichier .env
    with open(".env", "w") as f:
        f.write(f"BINANCE_API_KEY={api_key}\n")
        f.write(f"BINANCE_API_SECRET={api_secret}\n")
        f.write(f"USE_TESTNET={'True' if use_testnet else 'False'}\n")
    
    print("\nConfiguration sauvegardée dans le fichier .env")
    
    # Paramètres de trading personnalisés
    print("\n=== Paramètres de trading ===\n")
    print("Vous pouvez personnaliser les paramètres de trading ou utiliser les valeurs par défaut.")
    use_defaults = input("Utiliser les paramètres par défaut? (O/n): ").lower() != "n"
    
    if not use_defaults:
        try:
            risk_per_trade = float(input("Risque par trade (% du capital) [7.5]: ") or "7.5")
            stop_loss = float(input("Stop-loss (% du prix d'entrée) [4.0]: ") or "4.0")
            take_profit = float(input("Take-profit (% du prix d'entrée) [6.0]: ") or "6.0")
            leverage = int(input("Effet de levier [3]: ") or "3")
            
            # Créer un fichier de paramètres personnalisés
            params = {
                "RISK_PER_TRADE_PERCENT": risk_per_trade,
                "STOP_LOSS_PERCENT": stop_loss,
                "TAKE_PROFIT_PERCENT": take_profit,
                "LEVERAGE": leverage
            }
            
            with open("custom_params.json", "w") as f:
                json.dump(params, f, indent=2)
            
            print("\nParamètres personnalisés sauvegardés dans custom_params.json")
        except ValueError:
            print("Erreur: Valeur invalide. Utilisation des paramètres par défaut.")

def run_tests():
    """Exécute les tests unitaires"""
    print("\nExécution des tests unitaires...")
    
    try:
        import unittest
        
        if not os.path.exists("tests"):
            os.makedirs("tests")
            
        with open("tests/__init__.py", "w") as f:
            pass
        
        # Créer un test simple de connexion
        with open("tests/test_api_connection.py", "w") as f:
            f.write("""import unittest
import os
import sys

# Ajouter le répertoire parent au chemin de recherche
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.api_connector import BinanceConnector

class TestAPIConnection(unittest.TestCase):
    def test_connection(self):
        connector = BinanceConnector()
        self.assertTrue(connector.test_connection())

if __name__ == "__main__":
    unittest.main()
""")
        
        # Exécuter le test
        print("Test de connexion à l'API Binance...")
        result = subprocess.run([sys.executable, "-m", "unittest", "tests.test_api_connection"], capture_output=True)
        
        if result.returncode == 0:
            print("Test de connexion réussi")
        else:
            print("Test de connexion échoué. Vérifiez vos clés API.")
            print(result.stderr.decode())
    
    except Exception as e:
        print(f"Erreur lors de l'exécution des tests: {str(e)}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Installation du bot de trading")
    parser.add_argument("--skip-deps", action="store_true", help="Ignorer l'installation des dépendances")
    parser.add_argument("--skip-config", action="store_true", help="Ignorer la configuration")
    parser.add_argument("--skip-tests", action="store_true", help="Ignorer les tests")
    
    args = parser.parse_args()
    
    print("=== Installation du Bot de Trading Crypto ===\n")
    
    # Vérifier la version de Python
    check_python_version()
    
    # Créer les répertoires
    create_directories()
    
    # Installer les dépendances
    if not args.skip_deps:
        install_dependencies()
    
    # Configurer le bot
    if not args.skip_config:
        setup_configuration()
    
    # Exécuter les tests
    if not args.skip_tests:
        run_tests()
    
    print("\nInstallation terminée!")
    print("\nPour lancer le bot en mode test sans trading réel:")
    print("  python main.py --dry-run")
    print("\nPour lancer le bot en mode production:")
    print("  python main.py")
    print("\nPour exécuter un backtest:")
    print("  python backtest.py --symbol BTCUSDT --start 2023-01-01 --end 2023-06-30")

if __name__ == "__main__":
    main()