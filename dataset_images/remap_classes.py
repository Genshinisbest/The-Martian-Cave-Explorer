import os
import glob

# Old class numbers -> New class numbers mapping
# Based on your original classes.txt where:
# green_alien was line 15 -> should be 1
# stop_sign was line 16 -> should be 2  
# white_sphere was line 17 -> should be 5
# mushrooms was line 18 -> should be 3
# green_crystal was line 19 -> should be 0
# formation was line 20 -> should be 4

class_mapping = {
    '15': '1',  # green_alien
    '16': '2',  # stop_sign  
    '17': '5',  # white_sphere
    '18': '3',  # mushrooms
    '19': '0',  # green_crystal
    '20': '4'   # formation
}

# Process all .txt files
for txt_file in glob.glob("*.txt"):
    if txt_file == "classes.txt":
        continue  # Skip classes.txt
    
    print(f"Processing {txt_file}")
    
    # Read file
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    # Remap class numbers
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # Valid YOLO format line
            old_class = parts[0]
            if old_class in class_mapping:
                parts[0] = class_mapping[old_class]
                new_lines.append(' '.join(parts) + '\n')
                print(f"  Remapped class {old_class} -> {class_mapping[old_class]}")
            else:
                print(f"  Warning: Unknown class {old_class}")
        else:
            new_lines.append(line)  # Keep malformed lines as-is
    
    # Write back
    with open(txt_file, 'w') as f:
        f.writelines(new_lines)

print("Done! All annotation files updated.")
