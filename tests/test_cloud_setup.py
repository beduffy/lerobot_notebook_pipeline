#!/usr/bin/env python3
"""
Tests for cloud setup functionality - ensuring deployment scripts work correctly.
"""

import pytest
import tempfile
import subprocess
import os
import sys
from pathlib import Path
import importlib.util

def test_requirements_cloud_exists():
    """Test that requirements-cloud.txt exists and has essential packages."""
    req_file = Path("requirements-cloud.txt")
    assert req_file.exists(), "requirements-cloud.txt should exist"
    
    content = req_file.read_text()
    
    # Check for essential packages
    essential_packages = [
        "torch",
        "lerobot==0.2.0",
        "numpy",
        "matplotlib",
        "pytest"
    ]
    
    for package in essential_packages:
        assert package in content, f"Package {package} should be in requirements-cloud.txt"
    
    print("✅ requirements-cloud.txt contains essential packages")

def test_setup_cloud_py_syntax():
    """Test that setup-cloud.py has valid Python syntax."""
    setup_file = Path("setup-cloud.py")
    assert setup_file.exists(), "setup-cloud.py should exist"
    
    # Test syntax by attempting to compile
    try:
        with open(setup_file, 'r') as f:
            code = f.read()
        compile(code, setup_file, 'exec')
        print("✅ setup-cloud.py has valid Python syntax")
    except SyntaxError as e:
        pytest.fail(f"setup-cloud.py has syntax error: {e}")

def test_setup_cloud_sh_syntax():
    """Test that setup-cloud.sh exists and is executable."""
    setup_file = Path("setup-cloud.sh")
    assert setup_file.exists(), "setup-cloud.sh should exist"
    
    # Check if it's executable
    assert os.access(setup_file, os.X_OK), "setup-cloud.sh should be executable"
    
    # Basic shell syntax check
    result = subprocess.run(["bash", "-n", str(setup_file)], capture_output=True)
    assert result.returncode == 0, f"setup-cloud.sh has syntax errors: {result.stderr.decode()}"
    
    print("✅ setup-cloud.sh has valid syntax and is executable")

def test_pytest_ini_configuration():
    """Test that pytest.ini has correct configuration."""
    pytest_file = Path("pytest.ini")
    assert pytest_file.exists(), "pytest.ini should exist"
    
    content = pytest_file.read_text()
    
    # Check for essential configurations
    assert "--durations=" in content, "pytest.ini should configure test durations"
    assert "testpaths" in content, "pytest.ini should specify test paths"
    assert "markers" in content, "pytest.ini should define test markers"
    
    print("✅ pytest.ini configured correctly")

def test_cloud_setup_documentation():
    """Test that cloud setup documentation exists."""
    cloud_setup_file = Path("CLOUD_SETUP.md")
    quick_start_file = Path("QUICK_START.md")
    
    assert cloud_setup_file.exists(), "CLOUD_SETUP.md should exist"
    assert quick_start_file.exists(), "QUICK_START.md should exist"
    
    # Check content has essential information
    cloud_content = cloud_setup_file.read_text()
    assert "Lightning AI" in cloud_content, "Should mention Lightning AI"
    assert "pip install" in cloud_content, "Should have pip installation instructions"
    
    quick_content = quick_start_file.read_text()
    assert "setup-cloud.py" in quick_content, "Should reference setup script"
    
    print("✅ Cloud setup documentation exists and has essential content")

def test_package_installation_simulation():
    """Test package installation in isolation."""
    # Test if we can import the package structure
    try:
        import lerobot_notebook_pipeline
        from lerobot_notebook_pipeline.dataset_utils import analysis
        from lerobot_notebook_pipeline.dataset_utils import visualization
        print("✅ Package structure imports correctly")
    except ImportError as e:
        pytest.fail(f"Package import failed: {e}")

def test_lerobot_version_compatibility():
    """Test that LeRobot version is compatible."""
    try:
        import lerobot
        version = lerobot.__version__
        
        # Should be version 0.2.0 as specified in requirements
        expected_version = "0.2.0"
        assert version == expected_version, f"Expected LeRobot {expected_version}, got {version}"
        
        # Test that imports work with new structure
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.act.modeling_act import ACTPolicy
        
        print(f"✅ LeRobot {version} compatibility verified")
        
    except ImportError as e:
        pytest.fail(f"LeRobot compatibility test failed: {e}")

def test_cloud_environment_variables():
    """Test cloud environment variable handling."""
    # Test common cloud environment detection
    cloud_indicators = [
        "LIGHTNING_CLOUD_URL",
        "COLAB_GPU", 
        "KAGGLE_URL_BASE",
        "SAGEMAKER_DOMAIN"
    ]
    
    is_cloud = any(var in os.environ for var in cloud_indicators)
    
    if is_cloud:
        print("✅ Cloud environment detected")
    else:
        print("✅ Local environment (cloud variables not set)")

@pytest.mark.slow
def test_setup_script_dry_run():
    """Test setup script in dry-run mode."""
    setup_file = Path("setup-cloud.py")
    
    # Test that we can import the setup script
    spec = importlib.util.spec_from_file_location("setup_cloud", setup_file)
    setup_module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(setup_module)
        
        # Test that main function exists
        assert hasattr(setup_module, 'main'), "setup-cloud.py should have main() function"
        assert callable(setup_module.main), "main() should be callable"
        
        print("✅ Setup script module structure is correct")
        
    except Exception as e:
        pytest.fail(f"Setup script dry-run failed: {e}")

def test_gpu_availability_detection():
    """Test GPU/CUDA availability detection for cloud."""
    import torch
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA available with {device_count} device(s)")
    elif mps_available:
        print("✅ MPS (Apple Silicon) available")
    else:
        print("✅ CPU-only mode (no GPU detected)")
    
    # Test device selection logic
    if cuda_available:
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps") 
    else:
        device = torch.device("cpu")
    
    # Test tensor creation on selected device
    test_tensor = torch.randn(2, 2).to(device)
    assert test_tensor.device.type == device.type
    print(f"✅ Device selection works: {device}")

def test_dependency_conflicts():
    """Test for common dependency conflicts in cloud environments."""
    import pkg_resources
    
    # Get list of installed packages
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    # Check for critical packages
    critical_packages = ['torch', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in critical_packages:
        if package not in installed_packages:
            missing_packages.append(package)
    
    if missing_packages:
        pytest.fail(f"Missing critical packages: {missing_packages}")
    
    print("✅ Critical packages are installed")
    
    # Check for known problematic version combinations
    if 'torch' in installed_packages and 'torchvision' in installed_packages:
        torch_version = installed_packages['torch']
        torchvision_version = installed_packages['torchvision']
        print(f"✅ PyTorch ecosystem: torch={torch_version}, torchvision={torchvision_version}")

def test_memory_requirements():
    """Test basic memory requirements for cloud deployment."""
    import psutil
    
    # Get available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    
    # Should have at least 2GB available for basic operations
    assert available_gb >= 2.0, f"Insufficient memory: {available_gb:.1f}GB available"
    
    print(f"✅ Memory check passed: {available_gb:.1f}GB available")

def test_file_permissions():
    """Test that files have correct permissions for cloud deployment."""
    # Check that Python files are readable
    python_files = list(Path(".").glob("*.py"))
    for py_file in python_files[:5]:  # Check first 5 files
        assert os.access(py_file, os.R_OK), f"{py_file} should be readable"
    
    # Check that shell scripts are executable
    sh_files = list(Path(".").glob("*.sh"))
    for sh_file in sh_files:
        if sh_file.name == "setup-cloud.sh":
            assert os.access(sh_file, os.X_OK), f"{sh_file} should be executable"
    
    print("✅ File permissions are correct")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 