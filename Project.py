#John Brassil
#MA2071 Project


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


image_path = "Mushroom.webp"
image0 = Image.open(image_path)

image0_np = np.array(image0)
print("The dimension of the image_np is ", image0_np.shape)

plt.imshow(image0)
plt.title("Original Image (image0)")
plt.axis("off")
plt.show()

def image_transform(image_np, linear_transform):
  # Get the dimensions of the image
  height, width, channels = image_np.shape

  # Define the center
  center_x = width / 2
  center_y = height / 2

  # Loop through each pixel in the image and apply the transformation
  transformed_image = np.zeros_like(image_np)

  for y in range(height):
      for x in range(width):
          # Translate the pixel to the origin
          translated_x = x - center_x
          translated_y = -(y - center_y)

          # Apply the transformation: matrix vector multiplication
          transformed_x, transformed_y = linear_transform@np.array([translated_x, translated_y])

          # Translate the pixel back to its original position
          transformed_x += center_x
          transformed_y = - transformed_y + center_y

          # Round the pixel coordinates to integers
          transformed_x = int(round(transformed_x))
          transformed_y = int(round(transformed_y))

          # Copy the pixel to the transformed image
          if (transformed_x >= 0 and transformed_x < width and
              transformed_y >= 0 and transformed_y < height):
              transformed_image[transformed_y, transformed_x] = image_np[y, x]

  return transformed_image


# Transformation T1: Scaling by 0.5
T1 = np.array([[0.5, 0], [0, 0.5]])
image1_np = image_transform(image0_np, T1)
image1 = Image.fromarray(image1_np)
plt.imshow(image1)
plt.title("Scaled Image (image1)")
plt.axis("off")
plt.show()

# Transformation T2: Reflection through y = 2x
T2 = np.array([[-3/5, 4/5], [4/5, 3/5]])
image2_np = image_transform(image1_np, T2)
image2 = Image.fromarray(image2_np)
plt.imshow(image2)
plt.title("Reflected Image (image2)")
plt.axis("off")
plt.show()

# Transformation T3: Reflection through y = -1/2x
T3 = np.array([[3/5, -4/5], [-4/5, -3/5]])
image3_np = image_transform(image2_np, T3)
image3 = Image.fromarray(image3_np)
plt.imshow(image3)
plt.title("Reflected Image (image3)")
plt.axis("off")
plt.show()

# Transformation T: T2 o T1
T = np.dot(T2, T1)
image_T_np = image_transform(image1_np, T)
image_T = Image.fromarray(image_T_np)
plt.imshow(image_T)
plt.title("Composite Transformation T")
plt.axis("off")
plt.show()

# Inverse Transformation: Inverse of T
T_inv = np.linalg.inv(T)
image_T_inv_np = image_transform(image3_np, T_inv)
image_T_inv = Image.fromarray(image_T_inv_np)
plt.imshow(image_T_inv)
plt.title("Inverse Transformation T^-1")
plt.axis("off")
plt.show()



# Favorite Transformation: 30 degrees rotation, shear of 0.5
angle = np.radians(30)
shear_factor = 0.5

rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
shear_matrix = np.array([[1, shear_factor], [0, 1]])

favorite_transformation = np.dot(shear_matrix, rotation_matrix)
image4_np = image_transform(image0_np, favorite_transformation)
image4 = Image.fromarray(image4_np)
plt.imshow(image4)
plt.title("Favorite Transformation (image4)")
plt.axis("off")
plt.show()
