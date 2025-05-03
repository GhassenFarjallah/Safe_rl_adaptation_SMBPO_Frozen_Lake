# Safe_rl_adaptation_SMBPO_Frozen_Lake

#librairies utilisées: 
On a utilisé les librairies :
gym et gym_toy pour utiliser Frozen Lake

#les modifications apportées à l'algo pour fonctionner en environnement contenant espaces discret
De plus pour les parties code on a changé celle de SMBPO avec SAC en 2 réseaux Q parce que dans notre car les états dans  l'environnement est discret et non pas continue

On a utilisé le code existant de SMBPO (https://github.com/gwthomas/Safe-MBPO) pour seulement les parties  (rollout , https://github.com/Xingyu-Lin/mbpo_pytorch:

Éléments repris depuis gwthomas/Safe‑MBPO
•	Ensemble de dynamiques : MLP 64→64, entraîné 20×/ép.
•	Rollouts imaginés : simulate_rollout() pour augmenter le buffer.

Éléments repris depuis Xingyu-Lin/mbpo_pytorch
•	Architecture DynNet : trunk partagé + têtes séparées `next` (état) et `rew` (récompense).
•	Planification dynamique des rollouts via `ROLLOUT_SCHED` et fonction `rollout_horizon()`.
•	Entraînement de l’ensemble de dynamiques : `CrossEntropyLoss` pour états, MSE pour récompense.
•	Génération de rollouts synthétiques vectorisés en parallèle selon `ROLLOUTS_PER_STEP`.
•	Mix réel / modèle avec ratio `REAL_FRAC` pour les mises à jour SAC.
•	Soft-update (Polyak averaging) des réseaux Q-cibles (`q1_t`, `q2_t`).
•	Hyperparamètres et boucle MBPO (MODEL_ITERS, BATCH, ENSEMBLE_SIZE) calqués du dépôt.
•	Classes utilitaires : `Buffer` (add/sample), fonctions `set_seed`, `one_hot`, `to_tensor`.



