
import sys
import os

path = 'src/recon_lite/learning/m5_structure.py'
with open(path, 'rb') as f:
    content = f.read()

# Normalize to LF for easier replacement, then we'll save back (LF is usually safer in WSL)
lines = content.splitlines()
new_lines = []
skip = 0

for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    
    if b'# Check if ready for trial - EXPANSION: lowered to 0.40' in line:
        new_lines.append(b'            # Check if ready for trial - Delta-based analysis')
        new_lines.append(b'            consistency, signature, goal = cell.analyze_pattern_delta_based()')
        new_lines.append(b'            if goal:')
        new_lines.append(b'                cell.metadata["goal_vector"] = goal')
        skip = 1 # Skip the next line which is 'consistency, _ = cell.analyze_pattern()'
    else:
        new_lines.append(line)

with open(path, 'wb') as f:
    f.write(b'\n'.join(new_lines) + b'\n')

print("Fix applied successfully")
