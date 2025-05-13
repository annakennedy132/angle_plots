import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def point_to_line_distance(point, line):
    x0, y0 = point
    (x1, y1), (x2, y2) = line

    A = np.array([x1, y1])
    B = np.array([x2, y2])
    P = np.array([x0, y0])

    AB = B - A
    AP = P - A

    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return np.linalg.norm(AP), A

    t = np.dot(AP, AB) / AB_squared
    if t < 0:
        closest = A
    elif t > 1:
        closest = B
    else:
        closest = A + t * AB

    distance = np.linalg.norm(P - closest)
    return distance, closest

# --- Your data
corners = [(119,140), (750,158), (756,660), (100,653)]
point = [430,390]
image = "images/arena.tif"

# --- Load the image
img = mpimg.imread(image)

# --- Plotting
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')  # Show image in grayscale
ax.plot(*point, 'ro', label='Test Point')  # The point to measure from

# --- Plot corners as crosses
for corner in corners:
    ax.plot(corner[0], corner[1], 'bx', markersize=8, label='Corner' if corner == corners[0] else "")

# --- Draw rectangle and compute distances
min_distance = float('inf')
closest_line = None
closest_point_on_line = None

for i in range(len(corners)):
    p1 = corners[i]
    p2 = corners[(i+1) % len(corners)]  # Loop back to start
    # Draw rectangle edges
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')
    
    # Compute distance from point to this line
    dist, closest = point_to_line_distance(point, (p1, p2))
    ax.plot([point[0], closest[0]], [point[1], closest[1]], 'g--', alpha=0.6)
    
    # Store shortest
    if dist < min_distance:
        min_distance = dist
        closest_line = (p1, p2)
        closest_point_on_line = closest

# --- Highlight shortest distance
ax.plot(
    [point[0], closest_point_on_line[0]],
    [point[1], closest_point_on_line[1]],
    'r-', linewidth=2, label=f'Shortest Distance: {min_distance:.2f}'
)

ax.legend()
ax.set_title("Distance from Point to Rectangle Edges")
plt.show()
