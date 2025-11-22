# Guide de Déploiement Jenkins - GPU Service

## 📋 Prérequis

Avant de commencer, assurez-vous que:
- ✅ Votre code est sur GitHub: `https://github.com/MOHAMED-YASSIN-DERBEL/cuda-soa-lab`
- ✅ Tous les fichiers sont commités (Jenkinsfile, Dockerfile, main.py, etc.)
- ✅ Votre port étudiant est: **8115**

## 🚀 Étape 1: Pousser le Code sur GitHub

```powershell
# Vérifier le status
git status

# Ajouter tous les fichiers
git add .

# Commiter
git commit -m "feat: Complete Task 1-4 - GPU service with Jenkins CI/CD"

# Pousser sur GitHub
git push origin master
```

## 🔧 Étape 2: Créer le Pipeline Jenkins

### 2.1 Accéder à Jenkins
Ouvrez votre navigateur et allez à:
```
http://10.90.90.100:8090
```

### 2.2 Créer un Nouveau Job

1. **Cliquez sur "New Item"** (en haut à gauche)

2. **Configurer le job:**
   - **Name**: `gpu-lab-mohamed-yassin`
   - **Type**: Sélectionnez "Pipeline"
   - **Cliquez sur "OK"**

### 2.3 Configuration du Pipeline

#### General
- **Description**: 
  ```
  GPU Matrix Addition Service - Student: Mohamed Yassin Derbel
  Port: 8115
  ```

- ✅ **GitHub project**: 
  ```
  https://github.com/MOHAMED-YASSIN-DERBEL/cuda-soa-lab/
  ```

#### Build Triggers (Optionnel)
- ✅ **Poll SCM**: `H/5 * * * *`
  (Vérifie les changements toutes les 5 minutes)

#### Pipeline
- **Definition**: `Pipeline script from SCM`
- **SCM**: `Git`
- **Repository URL**: 
  ```
  https://github.com/MOHAMED-YASSIN-DERBEL/cuda-soa-lab.git
  ```
- **Credentials**: (Ajouter si repo privé)
- **Branch Specifier**: `*/master`
- **Script Path**: `Jenkinsfile`

### 2.4 Sauvegarder
Cliquez sur **"Save"**

## ▶️ Étape 3: Lancer le Build

1. **Cliquez sur "Build Now"** (menu gauche)

2. **Suivre la progression:**
   - Cliquez sur le numéro du build (ex: #1)
   - Cliquez sur "Console Output"
   - Regardez les logs en temps réel

## 📊 Étape 4: Vérifier le Déploiement

### 4.1 Vérifier l'État du Build

Dans la console Jenkins, vous devriez voir:
```
✅ GPU Sanity Test - OK
✅ Build Docker Image - OK
✅ Test Docker Image - OK
✅ Stop Old Container - OK
✅ Deploy Container - OK
✅ Health Check - OK
✅ Verify GPU Access - OK
```

### 4.2 Tester le Service Déployé

Depuis votre machine locale:

```powershell
# Test health endpoint
curl http://10.90.90.100:8115/health

# Test GPU info
curl http://10.90.90.100:8115/gpu-info

# Test documentation interactive
# Ouvrir dans le navigateur:
start http://10.90.90.100:8115/docs
```

### 4.3 Tester l'Addition de Matrices

```powershell
# Créer les matrices de test si nécessaire
python create_test_matrices.py

# Tester l'addition
curl -X POST "http://10.90.90.100:8115/add" `
  -F "file_a=@matrix1.npz" `
  -F "file_b=@matrix2.npz"
```

Réponse attendue:
```json
{
  "matrix_shape": [512, 512],
  "elapsed_time": 0.002134,
  "device": "GPU"
}
```

## 🔍 Étape 5: Monitoring et Logs

### 5.1 Voir les Logs du Container

Dans Jenkins console ou sur le serveur:
```bash
# Lister les containers
docker ps | grep gpu-service

# Voir les logs
docker logs gpu-service-<BUILD_NUMBER>

# Suivre les logs en temps réel
docker logs -f gpu-service-<BUILD_NUMBER>
```

### 5.2 Vérifier l'Utilisation GPU

```bash
# Depuis le serveur
nvidia-smi

# Ou via le container
docker exec gpu-service-<BUILD_NUMBER> nvidia-smi
```

## 🔄 Étape 6: Redéploiement Automatique

À chaque push sur GitHub:

```powershell
# Modifier votre code
# Par exemple, dans main.py

# Commiter et pousser
git add .
git commit -m "update: amélioration du service"
git push origin master

# Jenkins détectera automatiquement le changement
# et lancera un nouveau build
```

## 🐛 Dépannage

### Build Échoue - "GPU Sanity Test Failed"
**Cause**: L'agent Jenkins n'a pas de GPU  
**Solution**: Le pipeline continue quand même (warning only)

### Build Échoue - "Docker Build Failed"
**Vérifier**:
```bash
# Tester localement
docker build -t gpu-service:test .
```

### Build Échoue - "Health Check Failed"
**Causes possibles**:
1. Port 8115 déjà utilisé
2. Service trop long à démarrer
3. Erreur dans main.py

**Solution**:
```bash
# Vérifier les logs
docker logs gpu-service-<BUILD_NUMBER>

# Vérifier le port
docker ps | grep 8115
```

### Container Ne Démarre Pas
```bash
# Vérifier les erreurs
docker ps -a | grep gpu-service
docker logs gpu-service-<BUILD_NUMBER>

# Tester manuellement
docker run --gpus all -p 8115:8115 gpu-service:latest
```

### GPU Non Accessible
**Vérifier NVIDIA Container Toolkit**:
```bash
docker run --rm --gpus all nvidia/cuda:12.3.1-base nvidia-smi
```

## 📸 Captures d'Écran à Prendre

Pour votre rapport:

1. **Jenkins Dashboard** - Montrant votre job
2. **Build Console Output** - Build réussi
3. **Pipeline Stages** - Toutes les étapes en vert
4. **Service Response** - Résultat de curl health/gpu-info/add
5. **nvidia-smi Output** - GPU utilization pendant le test

## 🎯 Checklist Finale

Avant de considérer le déploiement complet:

- [ ] Repository GitHub à jour avec tous les fichiers
- [ ] Jenkinsfile configuré avec le bon port (8115)
- [ ] Pipeline Jenkins créé et configuré
- [ ] Build #1 réussi (tous les stages verts)
- [ ] Health endpoint répond: `http://10.90.90.100:8115/health`
- [ ] GPU info accessible: `http://10.90.90.100:8115/gpu-info`
- [ ] Addition de matrices fonctionne
- [ ] Container redémarre automatiquement
- [ ] Logs accessibles et propres

## 📝 Commandes Utiles

```bash
# Sur le serveur Jenkins (si vous avez accès)

# Lister tous vos containers
docker ps -a | grep gpu-service

# Arrêter un container
docker stop gpu-service-<BUILD_NUMBER>

# Supprimer un container
docker rm gpu-service-<BUILD_NUMBER>

# Voir les images Docker
docker images | grep gpu-service

# Nettoyer les anciennes images
docker image prune -f

# Vérifier l'espace disque
docker system df
```

## 🎓 Pour le Rapport

**Informations à inclure**:

1. **URL du Repository**: https://github.com/MOHAMED-YASSIN-DERBEL/cuda-soa-lab
2. **Nom du Job Jenkins**: gpu-lab-mohamed-yassin
3. **Port Assigné**: 8115
4. **URL du Service**: http://10.90.90.100:8115
5. **Technologies Utilisées**:
   - Python 3.11 + FastAPI
   - NVIDIA CUDA + Numba
   - Docker + NVIDIA Container Toolkit
   - Jenkins CI/CD
6. **Endpoints Implémentés**:
   - GET /health
   - GET /gpu-info
   - GET /gpu-load (bonus)
   - POST /add

## 🔗 Liens Utiles

- **Jenkins**: http://10.90.90.100:8090
- **Service**: http://10.90.90.100:8115
- **API Docs**: http://10.90.90.100:8115/docs
- **Prometheus** (Task 5): http://10.90.90.100:9090
- **Grafana** (Task 5): http://10.90.90.100:3000

## ✅ Statut des Tasks

- [x] Task 1: GPU Matrix Addition Service
- [x] Task 2: /gpu-info Endpoint
- [x] Task 3: Docker Containerization
- [x] Task 4: Jenkins CI/CD Pipeline
- [ ] Task 5: Prometheus Metrics & Grafana

## 🚀 Prochaine Étape

Une fois le déploiement Jenkins réussi, passez à:
**Task 5**: Ajout des métriques Prometheus et visualisation Grafana
