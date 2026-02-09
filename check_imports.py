
try:
    import bitwise_ops
    print("bitwise_ops imported successfully")
    print(dir(bitwise_ops))
except ImportError as e:
    print(f"Error importing bitwise_ops: {e}")

import models
print("models imported successfully")
