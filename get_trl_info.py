import trl
import inspect

print(f"TRL version: {trl.__version__}")
print(inspect.signature(trl.SFTTrainer))
