# Installation Guide for Python 3.14

## Issue
Python 3.14.1 is very new and `scikit-image` doesn't have pre-built wheels yet, requiring compilation from source.

## Solutions

### Option 1: Install Visual Studio Build Tools (Recommended)
This allows Python packages to compile from source on Windows.

**Steps:**
1. **Install Visual Studio Build Tools** (if not already done):
   - Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022
   - Run installer and select **"C++ build tools"** workload
   - Complete installation

2. **Use Developer Command Prompt** (IMPORTANT - regular PowerShell won't work):
   - Press `Win + S` and search for **"Developer Command Prompt for VS 2022"**
   - OR search for **"x64 Native Tools Command Prompt for VS 2022"**
   - Open it (this automatically sets up the compiler environment)

3. **In the Developer Command Prompt**, navigate to your project:
   ```cmd
   cd "C:\Users\aryan\OneDrive\Desktop\Projects\Claims Processor\health-insurance-claims-agent"
   .venv\Scripts\activate
   ```

4. **Install the packages**:
   ```cmd
   pip install scikit-image torch torchvision scipy opencv-python-headless
   ```

**Why Developer Command Prompt?** 
Regular PowerShell/CMD doesn't have the compiler in PATH. The Developer Command Prompt automatically activates the Visual Studio environment with all compiler tools.

**Note:** Your existing packages (numpy, Pillow, easyocr) will still be there - virtual environments persist!

### Option 2: Use Python 3.11 or 3.12 (Easier)
These versions have better wheel support. Create a new virtual environment:

```powershell
# Create new venv with Python 3.12 (if installed)
python3.12 -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install numpy Pillow scikit-image easyocr
```

### Option 3: Skip scikit-image (If not needed)
If your code doesn't directly use scikit-image, you can skip it. EasyOCR might work without it for basic functionality.

### Option 4: Use Conda (Alternative)
Conda has pre-built binaries for many packages:

```powershell
conda create -n claims-processor python=3.11
conda activate claims-processor
conda install -c conda-forge numpy pillow scikit-image easyocr
```

## Current Status
✅ numpy - Installed
✅ Pillow - Installed  
✅ easyocr - Installed (without dependencies)
✅ torch, torchvision, opencv-python-headless, scipy - Installing...
❌ scikit-image - Needs compilation or Visual Studio Build Tools

## Quick Test
After installing Visual Studio Build Tools, run:
```powershell
.venv\Scripts\python.exe -m pip install scikit-image
```

