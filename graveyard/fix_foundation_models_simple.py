#!/usr/bin/env python3
"""
Simple Foundation Model Dependency Checker

Avoids importing problematic packages and focuses on the core issue.
"""

import subprocess
import sys
from pathlib import Path

def check_package_versions():
    """Check package versions without importing them."""
    print("🔍 Checking Package Versions (Safe)")
    print("=" * 40)
    
    packages = ['numpy', 'transformers', 'torch', 'jax', 'jaxlib']
    
    for pkg in packages:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {pkg}; print({pkg}.__version__)'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ {pkg}: {version}")
            else:
                print(f"❌ {pkg}: Import failed")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()[:100]}...")
        except:
            print(f"⚠️  {pkg}: Check failed")
    
    return True

def test_foundation_models_isolated():
    """Test foundation models in isolated processes."""
    print("\n🧠 Testing Foundation Models (Isolated)")
    print("=" * 40)
    
    # Test SmolVLA without JAX interference
    smolvla_test = '''
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import warnings
warnings.filterwarnings("ignore")

try:
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    print("✅ SmolVLA Config: IMPORT OK")
    
    # Test basic config creation
    config = SmolVLAConfig(
        input_features={"observation.state": None, "observation.images.front": None},
        output_features={"action": None},
        use_quantization=False,
    )
    print("✅ SmolVLA Config: CREATION OK") 
    
except Exception as e:
    print(f"❌ SmolVLA Config: FAILED - {e}")

try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    print("✅ SmolVLA Model: IMPORT OK")
except Exception as e:
    print(f"❌ SmolVLA Model: FAILED - {e}")
'''
    
    # Test PI0Fast
    pi0fast_test = '''
import warnings
warnings.filterwarnings("ignore")

try:
    from lerobot.common.policies.pi0.configuration_pi0 import PI0Config  
    print("✅ PI0Fast Config: IMPORT OK")
    
    # Test basic config creation
    config = PI0Config(
        input_features={"observation.state": None, "observation.images.front": None},
        output_features={"action": None},
    )
    print("✅ PI0Fast Config: CREATION OK")
    
except Exception as e:
    print(f"❌ PI0Fast Config: FAILED - {e}")

try:
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    print("✅ PI0Fast Model: IMPORT OK")
except Exception as e:
    print(f"❌ PI0Fast Model: FAILED - {e}")
'''
    
    # Write and run tests
    tests = [
        ("SmolVLA", smolvla_test),
        ("PI0Fast", pi0fast_test)
    ]
    
    for name, test_code in tests:
        print(f"\n🧪 Testing {name}:")
        test_file = Path(f"test_{name.lower()}_isolated.py")
        
        with open(test_file, "w") as f:
            f.write(test_code)
        
        try:
            result = subprocess.run([sys.executable, str(test_file)], 
                                  capture_output=True, text=True, timeout=30)
            print(result.stdout)
            if result.stderr:
                stderr_clean = result.stderr.replace("/home/ben/miniconda3/envs/robosuite/lib/python3.10/site-packages/", "")
                print(f"🚨 Stderr: {stderr_clean[:200]}...")
        except subprocess.TimeoutExpired:
            print(f"⏰ {name} test timed out")
        except Exception as e:
            print(f"❌ {name} test failed: {e}")
        
        # Clean up
        test_file.unlink(missing_ok=True)

def provide_solution():
    """Provide the solution for foundation model issues."""
    print("\n🔧 SOLUTION FOR FOUNDATION MODEL ISSUES")
    print("=" * 50)
    
    print("🎯 ROOT CAUSE IDENTIFIED:")
    print("   JAX + NumPy version incompatibility causing SmolVLA imports to fail")
    print("   - NumPy 1.24.4 lacks 'dtypes' attribute that JAX 0.6.1 expects")
    print("   - This breaks the entire transformers → SmolVLA import chain")
    
    print("\n💡 RECOMMENDED FIXES:")
    
    print("\n📋 Option 1: Update NumPy (Recommended)")
    print("   conda activate robosuite")
    print("   pip install 'numpy>=2.0'")
    print("   # This should fix the JAX dtypes issue")
    
    print("\n📋 Option 2: Downgrade JAX")  
    print("   conda activate robosuite")
    print("   pip install 'jax<0.6.0' 'jaxlib<0.6.0'")
    print("   # Use older JAX compatible with NumPy 1.24")
    
    print("\n📋 Option 3: Remove JAX (If not needed)")
    print("   conda activate robosuite") 
    print("   pip uninstall jax jaxlib")
    print("   # SmolVLA might work without JAX")
    
    print("\n📋 Option 4: Create Clean Environment")
    print("   conda create -n foundation_models python=3.10")
    print("   conda activate foundation_models")
    print("   pip install torch torchvision")
    print("   pip install 'numpy>=2.0'")
    print("   pip install 'transformers>=4.36.0'")
    print("   pip install 'jax>=0.6.0' 'jaxlib>=0.6.0'")
    print("   # Install lerobot from source")
    
    print("\n🧪 Quick Test Command:")
    print("   # After applying fix:")
    print("   python test_model_inference.py --models smolvla,pi0")

def main():
    print("🛠️  FOUNDATION MODEL DEPENDENCY ANALYZER")
    print("=" * 50)
    
    check_package_versions()
    test_foundation_models_isolated()
    provide_solution()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Choose one of the fix options above")
    print("2. Apply the fix")
    print("3. Test with: python test_model_inference.py --models smolvla,pi0")
    print("4. If working, add to train_multi_model.py")

if __name__ == "__main__":
    main() 