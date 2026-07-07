import sys

print(sys.executable)

import sys

print("Executable:", sys.executable)
print("Version:", sys.version)

try:
    import torch
    print("Torch:", torch.__version__)
except Exception as e:
    print(e)