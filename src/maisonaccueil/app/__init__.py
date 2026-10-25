"""Lot B — Interface (PySide6), dans un processus distinct du traitement vidéo.

Organisation :

- ``ports``         : interfaces (Protocols) des dépôts injectés (profils, références,
                      réglages, journal). Le stockage du lot C les satisfait structurellement.
- ``messages``      : messages vision → interface en plus de ``VisionOutput`` et
                      ``CaptureState`` : aperçu réduit, état du moteur, échantillon de collecte.
- ``ipc``           : files bornées vision → interface (publication non bloquante côté
                      vision, lecture non bloquante côté interface).
- ``vision_worker`` : côté processus vision (capture + aperçu réduit), sans Qt.
- ``viewmodels``    : logique des écrans, testable SANS Qt.
- ``widgets``       : widgets PySide6 minces (seul paquet qui importe Qt Widgets).
- ``fakes``         : doubles ``Fake*`` pour développer sans caméra ni base.
- ``wiring``        : racine de composition (seul module qui importe ``maisonaccueil.storage``).
"""
