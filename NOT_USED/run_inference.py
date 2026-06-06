from carbontracker.tracker import CarbonTracker
import subprocess

# FOR TRACKING INFERENCE CARBON FOOTPRINT, AS CARBONTRACKER IS NATIVELY PYTHON
# AND DOES NOT SUPPORT C++ DIRECTLY!!
tracker = CarbonTracker(epochs=1)
tracker.epoch_start()

# Call the compiled C++ inference script
result = subprocess.run(["./inference"], capture_output=True, text=True)

# Print output from the C++ script
print(result.stdout)

tracker.epoch_end()
tracker.stop()
