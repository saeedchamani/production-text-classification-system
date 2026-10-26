import sys
from pathlib import Path


print("Python version:")
print(sys.version)

print("\nPython executable:")
print(sys.executable)

print("\nProject directory:")
print(Path.cwd())