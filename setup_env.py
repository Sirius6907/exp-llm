import sys
import os
import subprocess

def run_script(script_name):
    print(f"\n>>> Running {script_name}...")
    try:
        res = subprocess.run([sys.executable, script_name], check=True, capture_output=True, text=True)
        print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error during execution of {script_name}!")
        print(e.stderr)
        return False

def verify_environment():
    print("==================================================")
    print("      Environment Verification & Test Runner      ")
    print("==================================================")
    
    # 1. Check Python version
    print(f"Python Version: {sys.version}")
    
    # 2. Check PyTorch version and CUDA availability
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  - Device Capability: {torch.cuda.get_device_capability(0)}")
    except ImportError:
        print("[ERROR] PyTorch is not installed. Please run: pip install torch")
        sys.exit(1)
        
    # 3. Check for required module files
    required_files = ["mcp.py", "ssm_temporal.py", "tokenflow.py", "dynamap.py", "mock_test.py", "ssm_test.py", "orchestrator_test.py"]
    print("\nVerifying code components:")
    missing = False
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  - [Found] {filename}")
        else:
            print(f"  - [Missing] {filename}")
            missing = True
            
    if missing:
        print("[ERROR] One or more required pipeline modules are missing in this directory.")
        sys.exit(1)
    print("Success: All modular source files are present.")
    
    # 4. Execute test scripts sequentially
    success = True
    success &= run_script("mock_test.py")
    success &= run_script("ssm_test.py")
    success &= run_script("orchestrator_test.py")
    
    print("==================================================")
    if success:
        print("[SUCCESS] ALL TESTS PASSED! 3GB PIPELINE IS STABLE.")
    else:
        print("[FAIL] ONE OR MORE PIPELINE MODULES FAILED VALIDATION.")
    print("==================================================")
    
if __name__ == "__main__":
    verify_environment()
