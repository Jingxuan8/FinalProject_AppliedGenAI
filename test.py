from mcp.client.stdio import stdio_client

import inspect

print("Loaded object:", stdio_client)
print("Type:", type(stdio_client))

# Get the *source code* of the function
print("\n--- FUNCTION SOURCE ---")
print(inspect.getsource(stdio_client))

# Check what's inside the module where function is defined
print("\n--- MODULE MEMBERS ---")
mod = inspect.getmodule(stdio_client)
print("Module:", mod)
print("Module file:", mod.__file__)
print("Members:", [m for m in dir(mod) if "Client" in m])