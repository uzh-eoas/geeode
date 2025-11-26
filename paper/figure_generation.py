#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import textwrap

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Define the real value generating function
def f(x):
    return 2 * np.log(x + 1) + 3

# Create example population example member functions
def pm1(x):
    return 1.4 * np.log(x + 0.5) + 1.7
def pm2(x):
    return 2.4 * np.log(x + 1.5) + 4.2
def pm3(x):
    return 1.8 * np.log(x + 1.2) + 3.15

# Set the seed
np.random.seed(42)

# Produce the x values
x = np.linspace(0, 20, 200)

# Produce the y values
y_true = f(x)

# Then create their data
y_pm1 = pm1(x)
y_pm2 = pm2(x)
y_pm3 = pm3(x)

# Add random noise
noise = np.random.normal(loc=0.0, scale=0.3, size=x.shape) * 5
y_noisy = y_true + noise

# Create the underlying matplotlib figure
plt.figure(figsize=(8, 5))

# Scatterplot of the noisy data
plt.scatter(x, y_noisy, color='black', alpha=0.3,label='Randomized data')

# Overlay the clean function curve
plt.plot(x, y_true, color='black', linestyle=':', linewidth=2,label='Underlying Data Model')

# Overlay the population members
plt.plot(x, y_pm1, color='red', linewidth=2,label='Example Population Member - 1')
plt.plot(x, y_pm2, color='blue', linewidth=2,label='Example Population Member - 2')
plt.plot(x, y_pm3, color='green', linewidth=2,label='Example Population Member - 3')

# Finalize the other details of the chart
plt.xlim(0,20)
plt.ylim(0,12)
plt.title('Conceptual Example With Randomized Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True,alpha=0.75)
plt.legend()
plt.tight_layout()

# Display / save the plot
plt.savefig('main_figure.png', dpi=300)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Clear the figure then create the next figure
plt.clf()
plt.figure(figsize=(8, 5))

# Scatterplot of the noisy data
plt.scatter(x, y_noisy, color='black', alpha=0.3,label='Randomized data')

# Overlay the underlying data model
plt.plot(x, y_true, color='black', linestyle=':', linewidth=2,label='Underlying Data Model')

# Tidy up the figure (same style as the first one)
plt.xlim(0,20)
plt.ylim(0,12)
plt.title('Generated Randomized Data')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True,alpha=0.75)
plt.legend()
plt.tight_layout()

# Show (or save) the second plot
plt.savefig('generated_randomized_data.png', dpi=300)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Clear the figure then create the next figure
plt.clf()
plt.figure(figsize=(8, 5))

# Scatterplot of the noisy data
plt.scatter(x, y_noisy, color='black', alpha=0.3,label='Randomized data')

# Overlay the underlying data model
plt.plot(x, y_true, color='black', linestyle=':', linewidth=2,label='Underlying Data Model')

# Overlay the population members
plt.plot(x, y_pm3, color='green', linewidth=2,label='Example Population Member - 3')

# Tidy up the figure (same style as the first one)
plt.xlim(0,20)
plt.ylim(0,12)
plt.title('Best Model from Population')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True,alpha=0.75)
plt.legend()
plt.tight_layout()

# Show (or save) the second plot
plt.savefig('final_model.png', dpi=300)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Clear the figure then create the next figure
plt.clf()

# Make the circles
y = np.arange(0, 10, 0.5)
x = np.zeros_like(y)
mid = len(y) // 2

# Color them
bottom_colors = ['black' if i % 2 == 0 else 'blue' for i in range(mid)]
top_colors    = ['blue'] * (len(y) - mid)
colors = bottom_colors + top_colors   # order matches y

# Plot them
fig, ax = plt.subplots(figsize=(2, 6), dpi=100)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.scatter(x, y, c=colors, s=120)

# Adjust the plot
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.margins(x=0.2, y=0.05)

# Add the arrow
arrow_x   = 0.05
y_top     = y.max()
y_bottom  = y.min()

ax.annotate(
    '',
    xy=(arrow_x, y_top),
    xytext=(arrow_x, y_bottom),
    arrowprops=dict(
        arrowstyle='<->',
        linewidth=2,
        color='black'
    )
)

# Label it
def wrap_label(txt, width=12):
    return "\n".join(textwrap.wrap(txt, width=width))

low_label  = wrap_label("Lower Temporal Density",  width=12)
high_label = wrap_label("Higher Temporal Density", width=12)
label_offset = 0.04

bbox_props = dict(facecolor='none', edgecolor='none', pad=0.2)

ax.text(arrow_x + label_offset, y_top,
        low_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)

ax.text(arrow_x + label_offset, y_bottom,
        high_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)

# Show (or save) the second plot
plt.savefig('leapfrog_sampling.png', bbox_inches='tight', pad_inches=0)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Make the circles
y = np.arange(0, 10, 0.5)
x = np.zeros_like(y)

# Color them
num_red = 5
colors = ['black'] * num_red + ['blue'] * (len(y) - num_red)

# Plot them
fig, ax = plt.subplots(figsize=(2, 6), dpi=100)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')
ax.scatter(x, y, c=colors, s=120)

# Adjust the plot
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.margins(x=0.2, y=0.05)

# Add labels for the arrow
def wrap_label(txt, width=12):
    return "\n".join(textwrap.wrap(txt, width=width))

low_label  = wrap_label("Lower Temporal Density",  width=12)
high_label = wrap_label("Higher Temporal Density", width=12)

label_offset = 0.04
bbox_props = dict(facecolor='none', edgecolor='none', pad=0.2)

# Add the arrow
arrow_x   = 0.05
y_top     = y.max()
y_bottom  = y.min()

ax.annotate(
    '',
    xy=(arrow_x, y_top),
    xytext=(arrow_x, y_bottom),
    arrowprops=dict(
        arrowstyle='<->',
        linewidth=2,
        color='black'
    )
)

# Label it
def wrap_label(txt, width=12):
    return "\n".join(textwrap.wrap(txt, width=width))

low_label  = wrap_label("Lower Temporal Density",  width=12)
high_label = wrap_label("Higher Temporal Density", width=12)

label_offset = 0.04   # distance from the arrow line

bbox_props = dict(facecolor='none', edgecolor='none', pad=0.2)

ax.text(arrow_x + label_offset, y_top,
        low_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)

ax.text(arrow_x + label_offset, y_bottom,
        high_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)


# Save and show the figure
plt.savefig('bulk_sampling.png', bbox_inches='tight', pad_inches=0)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Make the circles
y = np.arange(0, 10, 0.6)
x = np.zeros_like(y)

# Color them
seq = ['red', 'green', 'blue', 'black']
colors = [seq[(len(y) - 1 - i) % len(seq)] for i in range(len(y))]

# Plot them
fig, ax = plt.subplots(figsize=(2, 6), dpi=100)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

scatter = ax.scatter(x, y, c=colors, s=120, edgecolor='none')

for idx, (xi, yi) in enumerate(zip(x, y)):
    rev_idx = len(y) - 1 - idx
    group_num = rev_idx // 4 + 1
    ax.text(
        xi, yi, str(group_num),
        ha='center', va='center',
       fontsize=8, color='white', fontweight='bold'
    )

# Adjust the plot
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.margins(x=0.2, y=0.05)

# Add the arrow
arrow_x   = 0.05
y_top     = y.max()
y_bottom  = y.min()

ax.annotate(
    '',
    xy=(arrow_x, y_top),
    xytext=(arrow_x, y_bottom),
    arrowprops=dict(
        arrowstyle='<->',          # double‑headed
        linewidth=2,
        color='black'
    )
)

# Add labels for the arrow
def wrap_label(txt, width=12):
    return "\n".join(textwrap.wrap(txt, width=width))

low_label  = wrap_label("Lower Temporal Density",  width=12)
high_label = wrap_label("Higher Temporal Density", width=12)

label_offset = 0.04   # distance from the arrow line

bbox_props = dict(facecolor='none', edgecolor='none', pad=0.2)

ax.text(arrow_x + label_offset, y_top,
        low_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)

ax.text(arrow_x + label_offset, y_bottom,
        high_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)


# Save and show the figure
plt.savefig('split_shuffle_sampling_pre.png', bbox_inches='tight', pad_inches=0)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Make the circles
y = np.arange(0, 10, 0.6)
x = np.zeros_like(y)

pattern = [
    ("red",   5),
  ("green", 4),
   ("blue",  4),
    ("black", 4),
]

# Color them
colours_top_first = []
labels_top_first = []

while len(colours_top_first) < len(y):
   for col, cnt in pattern:
     for i in range(cnt):
            if len(colours_top_first) >= len(y):
               break
            colours_top_first.append(col)
            labels_top_first.append(str(i + 1))


colors   = list(reversed(colours_top_first))
labels   = list(reversed(labels_top_first))

# Plot them
fig, ax = plt.subplots(figsize=(2, 6), dpi=100)
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

ax.scatter(x, y, c=colors, s=120, edgecolor="none")

# Add numbers inside the circles
for xi, yi, txt in zip(x, y, labels):
    ax.text(
       xi, yi, txt,
        ha="center", va="center",
        fontsize=8, color="white", fontweight="bold"
    )

# Adjust the plot
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.margins(x=0.2, y=0.05)

# Add the arrow
arrow_x   = 0.05
y_top     = y.max()
y_bottom  = y.min()

ax.annotate(
    '',
    xy=(arrow_x, y_top),
    xytext=(arrow_x, y_bottom),
    arrowprops=dict(
        arrowstyle='<->',          # double‑headed
        linewidth=2,
        color='black'
    )
)

# Add labels for the arrow
def wrap_label(txt, width=12):
    return "\n".join(textwrap.wrap(txt, width=width))

low_label  = wrap_label("Lower Temporal Density",  width=12)
high_label = wrap_label("Higher Temporal Density", width=12)

label_offset = 0.04   # distance from the arrow line


bbox_props = dict(facecolor='none', edgecolor='none', pad=0.2)

ax.text(arrow_x + label_offset, y_top,
        low_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)

ax.text(arrow_x + label_offset, y_bottom,
        high_label,
        va='center', ha='left',
        fontsize=11, fontweight='bold',
        wrap=True,
        bbox=bbox_props)

# Save and show the figure
plt.savefig('split_shuffle_sampling_post.png', bbox_inches='tight', pad_inches=0)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Generate some example data
np.random.seed(0)
t_base = np.arange(0, 10, 0.5)
y_base = np.sin(t_base) + 0.2 * np.random.randn(len(t_base))

n_extra = 50
t_extra = np.random.uniform(3, 7, size=n_extra)
y_extra = np.sin(t_extra) + 0.2 * np.random.randn(n_extra)

t = np.concatenate([t_base, t_extra])
y = np.concatenate([y_base, y_extra])

sort_idx = np.argsort(t)
t = t[sort_idx]
y = y[sort_idx]

# Highlight and example point
mid_idx = len(t) // 2
highlight_t = t[mid_idx]
highlight_y = y[mid_idx]

# Compute the ±σ kernel around the point
sigma_time = np.std(t)
left_bound  = highlight_t - sigma_time
right_bound = highlight_t + sigma_time


# Plot the data 
plt.figure(figsize=(11, 5))

# Single series of points (uniform colour & size)
plt.scatter(t, y, color='blue', edgecolor='black', s=60,
            label='Series')

# Vertical shaded region (±σ time window)
plt.axvspan(left_bound, right_bound, color='orange', alpha=0.2,
         label=r'$\pm\sigma$ time window')
plt.axvline(left_bound,  color='orange', linestyle='--', linewidth=1)
plt.axvline(right_bound, color='orange', linestyle='--', linewidth=1)

# Highlighted point (orange‑outlined circle)
plt.scatter(highlight_t, highlight_y,
            facecolor='none', edgecolor='orange', s=60,
            linewidth=2.5, zorder=5,
            label='Highlighted (mid point)')

# Finalize and save the plot
plt.title(f'Time Series Sampling Explained\n')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(loc='upper right', framealpha=1, facecolor='white')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ts_subsampling_explained.png', dpi=300)


