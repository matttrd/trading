from filters import _soft_pos_jumps
import random

def randomwalk1D(n):
    x, y = 0, 0
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = np.arange(n + 1)
    positions = [y]
    directions = ["UP", "DOWN"]
    for i in range(1, n + 1):
        # Randomly select either UP or DOWN
        step = random.choice(directions)
        
        # Move the object up or down
        if step == "UP":
            y += 1
        elif step == "DOWN":
            y -= 1
        # Keep track of the positions
        positions.append(y)
    return timepoints, positions


time_data, pos_data = randomwalk1D(1000)


threshold = 1/100
out = _soft_pos_jumps(np.array(pos_data, dtype=np.float32) , 20, threshold, perc=True)