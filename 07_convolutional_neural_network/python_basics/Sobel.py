"""
The Sobel operator is one of the simplest ways to explain edge detection in images — something middle school students can easily relate to when thinking about “outlines” in drawings.

What is the Sobel Operator?

Think of it like a special magnifying glass 🔍 for images.

It looks at each pixel and compares it with its neighbors.

If there’s a big change in brightness (like from white → black), it marks it as an edge.

If the brightness is smooth (all same color), it does nothing.

👉 Real-world: Banks and financial institutions use this in check processing 🏦 to read handwritten signatures, account numbers, or in fraud detection when scanning altered documents.
"""
import cv2
import matplotlib.pyplot as plt

# 1) Load a sample image in grayscale (black and white)
# For teaching: a simple shape (you can replace "car.png" with any image)
image = cv2.imread(cv2.samples.findFile("lena.png"), cv2.IMREAD_GRAYSCALE)

# 2) Apply Sobel operator in X direction (vertical edges)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

# 3) Apply Sobel operator in Y direction (horizontal edges)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 4) Combine both directions
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# 5) Show results
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Edges (X)")
plt.imshow(sobel_x, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Edges (Y)")
plt.imshow(sobel_y, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Combined Edges")
plt.imshow(sobel_combined, cmap='gray')
plt.axis('off')

plt.show()


"""
How to Teach This
    Concept: Like tracing outlines in a coloring book 🖍️.
    Purpose: Detect edges in images (faces, objects, handwriting).
    Activity: Let students try with pictures of their school logo, handwritten names, or shapes.
    Outcome: They’ll see how a computer “finds the edges” automatically.

Sobel X: finds up-down edges (changes left↔right).
Sobel Y: finds left-right edges (changes up↔down).
Magnitude: combines both—strong edges glow brightest.
    
"""
