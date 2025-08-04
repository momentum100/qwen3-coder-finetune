#!/usr/bin/env python3
import csv

# Read the current file and fix any issues
data = []
with open('animal_sounds_dataset.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# First line should be the header
if not lines[0].startswith('instruction,output'):
    lines.insert(0, 'instruction,output\n')

# Write properly formatted CSV
with open('animal_sounds_dataset_fixed.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['instruction', 'output'])
    
    for line in lines[1:]:
        if line.strip():
            # Parse the line manually to handle quotes
            parts = line.strip().split('","')
            if len(parts) == 2:
                instruction = parts[0].strip('"')
                output = parts[1].strip('"')
                writer.writerow([instruction, output])

print("Fixed CSV saved to animal_sounds_dataset_fixed.csv")

# Replace the original
import shutil
shutil.move('animal_sounds_dataset_fixed.csv', 'animal_sounds_dataset.csv')
print("Original file updated")