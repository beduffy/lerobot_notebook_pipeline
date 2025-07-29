#!/usr/bin/env python3
"""
Foundation Model Dependency Fixer

Diagnoses and fixes dependency issues for SmolVLA and PI0Fast.
The main issues are:
1. SmolVLA: numpy.dtypes AttributeError with transformers
2. PI0Fast: Missing GemmaForCausalLM import

Usage:
    python fix_foundation_models.py --check-deps
    python fix_foundation_models.py --test-smolvla  
    python fix_foundation_models.py --test-pi0fast
"""

import argparse
import subprocess
import sys
import importlib
import warnings
from pathlib import Path

def check_dependencies():
    """Check and report dependency versions."""
    print("🔍 Checking Foundation Model Dependencies")
    print("=" * 50)
    
    # Key packages for foundation models
    packages = [
        'numpy', 'transformers', 'torch', 'torchvision', 
        'jax', 'tensorflow', 'huggingface_hub'
    ]
    
    versions = {}
    for pkg in packages:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, '__version__', 'unknown')
            versions[pkg] = version
            print(f"✅ {pkg}: {version}")
        except ImportError:
            versions[pkg] = 'NOT INSTALLED'
            print(f"❌ {pkg}: NOT INSTALLED")
    
    # Check for known problematic combinations
    print("\n🔍 Dependency Analysis:")
    
    if 'numpy' in versions and versions['numpy'] != 'NOT INSTALLED':
        numpy_version = versions['numpy']
        if numpy_version.startswith('2.'):
            print("⚠️  NumPy 2.x detected - may cause issues with older transformers")
            print("   Recommendation: Consider downgrading to numpy<2.0")
    
    if 'transformers' in versions and versions['transformers'] != 'NOT INSTALLED':
        tf_version = versions['transformers']
        print(f"📦 Transformers version: {tf_version}")
    
    return versions

def test_smolvla_imports():
    """Test SmolVLA imports step by step."""
    print("\n🧠 Testing SmolVLA Imports")
    print("=" * 30)
    
    # Test imports one by one
    imports_to_test = [
        ("torch", "Basic PyTorch"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers (basic)"),
        ("transformers.AutoProcessor", "AutoProcessor"),
        ("lerobot.common.policies.smolvla.configuration_smolvla", "SmolVLA Config"),
        ("lerobot.common.policies.smolvla.modeling_smolvla", "SmolVLA Model")
    ]
    
    results = {}
    
    for import_name, description in imports_to_test:
        try:
            if "." in import_name:
                # Handle from X import Y syntax
                parts = import_name.split(".")
                module_name = ".".join(parts[:-1])
                attr_name = parts[-1]
                module = importlib.import_module(module_name)
                getattr(module, attr_name)
            else:
                importlib.import_module(import_name)
            
            print(f"✅ {description}: OK")
            results[import_name] = "OK"
            
        except Exception as e:
            print(f"❌ {description}: FAILED")
            print(f"   Error: {str(e)[:100]}...")
            results[import_name] = str(e)
    
    return results

def test_pi0fast_imports():
    """Test PI0Fast imports step by step."""
    print("\n🥧 Testing PI0Fast Imports")
    print("=" * 30)
    
    imports_to_test = [
        ("torch", "Basic PyTorch"),
        ("transformers", "Transformers"),
        ("transformers.GemmaForCausalLM", "GemmaForCausalLM"),
        ("lerobot.common.policies.pi0.configuration_pi0", "PI0 Config"),
        ("lerobot.common.policies.pi0.modeling_pi0", "PI0 Model")
    ]
    
    results = {}
    
    for import_name, description in imports_to_test:
        try:
            if "." in import_name:
                parts = import_name.split(".")
                module_name = ".".join(parts[:-1])
                attr_name = parts[-1]
                module = importlib.import_module(module_name)
                getattr(module, attr_name)
            else:
                importlib.import_module(import_name)
            
            print(f"✅ {description}: OK")
            results[import_name] = "OK"
            
        except Exception as e:
            print(f"❌ {description}: FAILED")
            print(f"   Error: {str(e)[:100]}...")
            results[import_name] = str(e)
    
    return results

def create_simple_smolvla_test():
    """Create a simplified SmolVLA test that bypasses problematic imports."""
    print("\n🛠️ Creating Simplified SmolVLA Test")
    print("=" * 40)
    
    test_code = '''
import torch
import warnings
warnings.filterwarnings("ignore")

# Try to create a minimal SmolVLA-like configuration
try:
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    
    # Minimal config for testing
    config = SmolVLAConfig(
        input_features={},
        output_features={},
        use_quantization=False,
    )
    print("✅ SmolVLA Config creation: SUCCESS")
    
except Exception as e:
    print(f"❌ SmolVLA Config creation: FAILED - {e}")

# Try model creation with mock data
try:
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    # This might fail due to dependency issues, but we can catch it
    print("⚠️  SmolVLA Model import available, but may have runtime issues")
    
except Exception as e:
    print(f"❌ SmolVLA Model import: FAILED - {e}")
'''
    
    # Write test file
    test_file = Path("test_smolvla_simple.py")
    with open(test_file, "w") as f:
        f.write(test_code)
    
    print(f"📝 Created test file: {test_file}")
    
    # Run the test
    try:
        result = subprocess.run([sys.executable, str(test_file)], 
                              capture_output=True, text=True, timeout=30)
        print("\n📊 Test Results:")
        print(result.stdout)
        if result.stderr:
            print("🚨 Errors:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")

def suggest_fixes(smolvla_results, pi0fast_results):
    """Suggest fixes based on test results."""
    print("\n🔧 RECOMMENDED FIXES")
    print("=" * 30)
    
    # SmolVLA fixes
    if any("numpy" in str(result) for result in smolvla_results.values()):
        print("🧠 SmolVLA NumPy Issue:")
        print("   1. Try downgrading numpy: pip install 'numpy<2.0'")
        print("   2. Update transformers: pip install --upgrade transformers")
        print("   3. Clear cache: rm -rf ~/.cache/huggingface/")
    
    if any("AutoProcessor" in str(result) for result in smolvla_results.values()):
        print("🧠 SmolVLA AutoProcessor Issue:")
        print("   1. Update transformers: pip install transformers>=4.30.0")
        print("   2. Install specific version: pip install transformers==4.36.0")
    
    # PI0Fast fixes  
    if any("GemmaForCausalLM" in str(result) for result in pi0fast_results.values()):
        print("🥧 PI0Fast GemmaForCausalLM Issue:")
        print("   1. Update transformers: pip install --upgrade transformers")
        print("   2. Install latest: pip install transformers>=4.38.0")
        print("   3. Check if Gemma model is available in your transformers version")
    
    # General fixes
    print("\n🔄 General Foundation Model Setup:")
    print("   1. Create clean environment:")
    print("      conda create -n foundation_models python=3.10")
    print("      conda activate foundation_models")
    print("   2. Install in correct order:")
    print("      pip install torch torchvision")
    print("      pip install 'numpy<2.0'")  
    print("      pip install transformers>=4.36.0")
    print("      pip install huggingface_hub")
    print("   3. Install lerobot last:")
    print("      pip install -e /path/to/lerobot")

def main():
    parser = argparse.ArgumentParser(description="Fix foundation model dependency issues")
    parser.add_argument("--check-deps", action="store_true", help="Check dependency versions")
    parser.add_argument("--test-smolvla", action="store_true", help="Test SmolVLA imports")
    parser.add_argument("--test-pi0fast", action="store_true", help="Test PI0Fast imports")
    parser.add_argument("--suggest-fixes", action="store_true", help="Suggest fixes")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all or not any([args.check_deps, args.test_smolvla, args.test_pi0fast, args.suggest_fixes]):
        args.check_deps = True
        args.test_smolvla = True  
        args.test_pi0fast = True
        args.suggest_fixes = True
    
    smolvla_results = {}
    pi0fast_results = {}
    
    if args.check_deps:
        versions = check_dependencies()
    
    if args.test_smolvla:
        smolvla_results = test_smolvla_imports()
        create_simple_smolvla_test()
    
    if args.test_pi0fast:
        pi0fast_results = test_pi0fast_imports()
    
    if args.suggest_fixes:
        suggest_fixes(smolvla_results, pi0fast_results)
    
    print("\n🎯 Next Steps:")
    print("1. Apply recommended fixes")
    print("2. Test with: python test_model_inference.py --models smolvla,pi0")
    print("3. Once working, add to train_multi_model.py")

if __name__ == "__main__":
    main() 