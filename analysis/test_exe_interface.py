
# Quick test script to verify the lab_proj executable interface
import subprocess
import sys
from pathlib import Path

def test_executable():
    exe_path = Path("D:/App External Data/Projects/VUB-ACS-25_GPU/bin/lab_proj.exe")

    if not exe_path.exists():
        print(f"❌ Executable not found: {exe_path}")
        return False

    print(f"✅ Found executable: {exe_path}")

    # Test with no arguments to see usage
    print("\n📝 Testing usage message:")
    try:
        result = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=10)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Return code:", result.returncode)
    except Exception as e:
        print(f"Error: {e}")

    # Test with 3 arguments
    print("\n🧪 Testing with 3 arguments (sum 10000 10):")
    try:
        result = subprocess.run([str(exe_path), "sum", "10000", "10"],
                              capture_output=True, text=True, timeout=30)
        print("Return code:", result.returncode)
        if result.returncode == 0:
            print("✅ 3-argument format works!")
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print("❌ 3-argument format failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")

    # Test with 4 arguments
    print("\n🧪 Testing with 4 arguments (sum 10000 10 32):")
    try:
        result = subprocess.run([str(exe_path), "sum", "10000", "10", "32"],
                              capture_output=True, text=True, timeout=30)
        print("Return code:", result.returncode)
        if result.returncode == 0:
            print("✅ 4-argument format works!")
            print("Output:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
        else:
            print("❌ 4-argument format failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_executable()
