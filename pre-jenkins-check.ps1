# pre-jenkins-check.ps1 - Vérification avant déploiement Jenkins

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Vérification Pré-Déploiement Jenkins" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$allChecks = $true

# Vérification 1: Fichiers requis
Write-Host "1. Vérification des fichiers requis..." -ForegroundColor Yellow
$requiredFiles = @(
    "main.py",
    "Dockerfile",
    "Jenkinsfile",
    "cuda_test.py",
    "requirements.txt",
    "pyproject.toml"
)

foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "   ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "   ✗ $file - MANQUANT!" -ForegroundColor Red
        $allChecks = $false
    }
}

# Vérification 2: Configuration du port dans Jenkinsfile
Write-Host "`n2. Vérification du port dans Jenkinsfile..." -ForegroundColor Yellow
$jenkinsContent = Get-Content "Jenkinsfile" -Raw
if ($jenkinsContent -match 'STUDENT_PORT = "(\d+)"') {
    $port = $matches[1]
    Write-Host "   ✓ Port configuré: $port" -ForegroundColor Green
    
    # Vérifier cohérence avec main.py
    $mainContent = Get-Content "main.py" -Raw
    if ($mainContent -match "STUDENT_PORT = (\d+)") {
        $mainPort = $matches[1]
        if ($port -eq $mainPort) {
            Write-Host "   ✓ Port cohérent dans main.py" -ForegroundColor Green
        } else {
            Write-Host "   ⚠ Port différent dans main.py: $mainPort" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "   ✗ Port non configuré dans Jenkinsfile!" -ForegroundColor Red
    $allChecks = $false
}

# Vérification 3: Git repository
Write-Host "`n3. Vérification du repository Git..." -ForegroundColor Yellow
if (Test-Path ".git") {
    Write-Host "   ✓ Repository Git initialisé" -ForegroundColor Green
    
    # Vérifier l'origine
    $remote = git remote get-url origin 2>$null
    if ($remote) {
        Write-Host "   ✓ Remote: $remote" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ Remote origin non configuré" -ForegroundColor Yellow
    }
    
    # Vérifier les fichiers non commités
    $status = git status --porcelain
    if ($status) {
        Write-Host "   ⚠ Fichiers non commités:" -ForegroundColor Yellow
        $status | ForEach-Object { Write-Host "     $_" -ForegroundColor Yellow }
    } else {
        Write-Host "   ✓ Tous les fichiers sont commités" -ForegroundColor Green
    }
} else {
    Write-Host "   ✗ Pas de repository Git!" -ForegroundColor Red
    $allChecks = $false
}

# Vérification 4: Docker
Write-Host "`n4. Vérification de Docker..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($dockerVersion) {
        Write-Host "   ✓ Docker installé: $dockerVersion" -ForegroundColor Green
    } else {
        Write-Host "   ⚠ Docker non trouvé (optionnel pour test local)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠ Docker non disponible" -ForegroundColor Yellow
}

# Vérification 5: Python et dépendances
Write-Host "`n5. Vérification de l'environnement Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "   ✓ Python: $pythonVersion" -ForegroundColor Green
    
    # Vérifier les packages critiques
    $criticalPackages = @("fastapi", "numba", "numpy", "uvicorn")
    foreach ($package in $criticalPackages) {
        $installed = python -c "import $package; print('OK')" 2>$null
        if ($installed -eq "OK") {
            Write-Host "   ✓ Package $package installé" -ForegroundColor Green
        } else {
            Write-Host "   ✗ Package $package manquant!" -ForegroundColor Red
            $allChecks = $false
        }
    }
} catch {
    Write-Host "   ✗ Python non trouvé!" -ForegroundColor Red
    $allChecks = $false
}

# Vérification 6: Test rapide du code
Write-Host "`n6. Test de syntaxe Python..." -ForegroundColor Yellow
try {
    $syntaxCheck = python -m py_compile main.py 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "   ✓ main.py syntaxe OK" -ForegroundColor Green
    } else {
        Write-Host "   ✗ Erreur de syntaxe dans main.py!" -ForegroundColor Red
        Write-Host "     $syntaxCheck" -ForegroundColor Red
        $allChecks = $false
    }
} catch {
    Write-Host "   ⚠ Impossible de vérifier la syntaxe" -ForegroundColor Yellow
}

# Vérification 7: Dockerfile
Write-Host "`n7. Vérification du Dockerfile..." -ForegroundColor Yellow
$dockerfileContent = Get-Content "Dockerfile" -Raw
if ($dockerfileContent -match "FROM nvidia/cuda") {
    Write-Host "   ✓ Base image CUDA correcte" -ForegroundColor Green
} else {
    Write-Host "   ⚠ Base image CUDA non standard" -ForegroundColor Yellow
}

if ($dockerfileContent -match "EXPOSE.*8001") {
    Write-Host "   ✓ Port exposé dans Dockerfile" -ForegroundColor Green
} else {
    Write-Host "   ⚠ Port non exposé dans Dockerfile" -ForegroundColor Yellow
}

# Résumé final
Write-Host "`n========================================" -ForegroundColor Cyan
if ($allChecks) {
    Write-Host "✅ TOUT EST PRÊT POUR JENKINS!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nProchaines étapes:" -ForegroundColor White
    Write-Host "1. git add ." -ForegroundColor White
    Write-Host "2. git commit -m 'feat: Complete GPU service implementation'" -ForegroundColor White
    Write-Host "3. git push origin master" -ForegroundColor White
    Write-Host "4. Créer le job Jenkins: http://10.90.90.100:8090" -ForegroundColor White
    Write-Host "5. Lancer 'Build Now'" -ForegroundColor White
} else {
    Write-Host "❌ PROBLÈMES DÉTECTÉS!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "`nCorrigez les erreurs avant de déployer sur Jenkins." -ForegroundColor Yellow
}
Write-Host ""
