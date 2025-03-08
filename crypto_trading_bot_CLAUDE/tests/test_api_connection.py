import unittest
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
