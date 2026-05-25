# Safe_rl_adaptation_SMBPO_Frozen_Lake

### 📺 Démonstration Vidéo
Voici une visualisation de l'exécution de l'algorithme sur l'environnement Frozen Lake :

[Adapted SMBPO for Frozen Lake](https://github.com/GhassenFarjallah/Safe_rl_adaptation_SMBPO_Frozen_Lake/blob/main/video_Adaptation_SMBPO.mp4)>

---

### 📚 Librairies utilisées
On a utilisé les librairies : `gym` et `gym_toy` pour utiliser Frozen Lake.

### ⚙️ Modifications apportées à l'algorithme
* **Adaptation aux espaces discrets :** Pour les parties code, on a changé celle de SMBPO avec SAC en **2 réseaux Q** parce que dans notre cas, les états dans l'environnement sont discrets et non continus.

---

### 🧬 Origine du Code (Reprises et Crédits)
On a utilisé le code existant de [gwthomas/Safe-MBPO](https://github.com/gwthomas/Safe-MBPO) et de [Xingyu-Lin/mbpo_pytorch](https://github.com/Xingyu-Lin/mbpo_pytorch) pour les composants suivants :

#### 🔹 Éléments repris depuis `gwthomas/Safe‑MBPO`
* **Ensemble de dynamiques :** MLP 64→64, entraîné 20×/épisode.
* **Rollouts imaginés :** Fonction `simulate_rollout()` pour augmenter le buffer.

#### 🔹 Éléments repris depuis `Xingyu-Lin/mbpo_pytorch`
* **Architecture DynNet :** Trunk partagé + têtes séparées `next` (état) et `rew` (récompense).
* **Planification dynamique des rollouts :** Via `ROLLOUT_SCHED` et la fonction `rollout_horizon()`.
* **Entraînement de l’ensemble de dynamiques :** `CrossEntropyLoss` pour les états, `MSE` pour la récompense.
* **Génération de rollouts synthétiques :** Vectorisés en parallèle selon `ROLLOUTS_PER_STEP`.
* **Mix réel / modèle :** Utilisation du ratio `REAL_FRAC` pour les mises à jour SAC.
* **Soft-update :** Polyak averaging des réseaux Q-cibles (`q1_t`, `q2_t`).
* **Boucle MBPO :** Hyperparamètres et itérations (`MODEL_ITERS`, `BATCH`, `ENSEMBLE_SIZE`) calqués du dépôt.
* **Classes utilitaires :** `Buffer` (add/sample), fonctions `set_seed`, `one_hot`, `to_tensor`.

---

### 🚀 Exécution du code

* **`SMBPO_with_cost_obstacles.py`** : Vous pouvez exécuter en local sur Spyder ou Jupyter Notebook le code de SMBPO modifié avec le coût relatif. Il contient les deux parties : simulation (train) et réalité (test).
* **`MBPO_On_FrozenLake_Env.py`** : Vous pouvez exécuter le code de la méthode non sécuritaire MBPO avec les environnements train et test ainsi que les visualisations.
