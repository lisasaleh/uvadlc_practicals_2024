import re
import matplotlib.pyplot as plt

# Path to the SLURM output file
slurm_file = "slurm_output_8658383.out"

# Regular expressions to extract relevant information
conv_type_regex = re.compile(r"Type of convolution :\s+(\w+)")
val_mean_regex = re.compile(r"mean: ([\d.]+) std: [\d.]+ for validation")
test_mean_regex = re.compile(r"mean: ([\d.]+) std: [\d.]+ for test")

# Extract data
conv_types = []
val_means = []
test_means = []

with open(slurm_file, 'r') as f:
    lines = f.read()
    conv_types = conv_type_regex.findall(lines)
    val_means = [float(m) for m in val_mean_regex.findall(lines)]
    test_means = [float(m) for m in test_mean_regex.findall(lines)]

# Prepare table data
table_data = [["Convolution Type", "Validation Mean", "Test Mean"]]
for conv, val, test in zip(conv_types, val_means, test_means):
    table_data.append([conv, f"{val:.4f}", f"{test:.4f}"])

# Create a figure for the table
fig, ax = plt.subplots(figsize=(8, len(table_data) * 0.6))
ax.axis('off')  # Turn off the axis

# Create the table
table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(table_data[0]))))

# Save the table as a PNG
output_file = "ex1d.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Table saved as {output_file}")

# Show the table (optional)
plt.show()
