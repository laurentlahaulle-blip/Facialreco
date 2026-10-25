"""Widgets PySide6. Seul paquet de l'interface qui importe Qt Widgets.

Ne pas importer ce paquet sans vérifier ``qt_available()`` (``app.main``) : sur une machine
sans bibliothèques graphiques (ex. ``libEGL`` absent), l'import de ``QtWidgets`` échoue.
"""
