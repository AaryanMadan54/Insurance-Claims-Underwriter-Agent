# Script to install packages with Visual Studio Build Tools activated
# This activates the VS environment and then installs packages

Write-Host "Activating Visual Studio Build Tools environment..." -ForegroundColor Yellow

# Try to find and activate VS environment
$vsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

$vsPath = $null
foreach ($path in $vsPaths) {
    if (Test-Path $path) {
        $vsPath = $path
        break
    }
}

if ($vsPath) {
    Write-Host "Found VS at: $vsPath" -ForegroundColor Green
    # Activate VS environment using cmd (PowerShell can't directly source .bat files)
    cmd /c "`"$vsPath`" && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
        }
    }
    Write-Host "VS environment activated!" -ForegroundColor Green
} else {
    Write-Host "WARNING: Could not find Visual Studio Build Tools automatically." -ForegroundColor Red
    Write-Host "Please use 'Developer Command Prompt for VS 2022' instead:" -ForegroundColor Yellow
    Write-Host "1. Search for 'Developer Command Prompt for VS 2022' in Start Menu" -ForegroundColor Yellow
    Write-Host "2. Run it, then navigate to your project and activate venv" -ForegroundColor Yellow
    Write-Host "3. Run: pip install scikit-image torch torchvision scipy opencv-python-headless" -ForegroundColor Yellow
    exit 1
}

# Navigate to project directory
$projectDir = "C:\Users\aryan\OneDrive\Desktop\Projects\Claims Processor\health-insurance-claims-agent"
Set-Location $projectDir

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Yellow
& "$projectDir\.venv\Scripts\Activate.ps1"

# Install packages
Write-Host "`nInstalling packages..." -ForegroundColor Yellow
pip install scikit-image torch torchvision scipy opencv-python-headless

Write-Host "`nDone!" -ForegroundColor Green


