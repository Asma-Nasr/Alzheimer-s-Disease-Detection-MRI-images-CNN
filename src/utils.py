from PIL import Image
import matplotlib.pyplot as plt

def display_image(image_path, grayscale=False):
    """Display image with pixel-perfect tight layout

    Args:
        image_path (str): Path to the image file
        grayscale (bool): If True, displays image in grayscale. Default is False.
    """
    img = Image.open(image_path)
    if grayscale:
        img = img.convert('L')  # Convert to grayscale
    
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Use 'gray' colormap if grayscale, otherwise use default RGB
    cmap = 'gray' if grayscale else None
    ax.imshow(img, aspect='auto', cmap=cmap)
    plt.show()
