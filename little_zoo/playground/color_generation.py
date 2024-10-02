import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box

class Color:
    def __init__(self, color):
        """
        Implements a color class characterized by a color and shade attributes.
        Parameters
        ----------
        color: str
            Color in red, blue, green.
        shade: str
            Shade in light, dark.
        """
        self.color = color
        
        # Define the colors and their corresponding RGB values
        self.colors_rgb = {
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0]),
            "blue": np.array([0, 0, 255]),
            "yellow": np.array([255, 255, 0]),
            "purple": np.array([128, 0, 128]),
            "orange": np.array([255, 165, 0]),
            "pink": np.array([255, 192, 203]),
            "brown": np.array([165, 42, 42]),
            "black": np.array([0, 0, 0]),
            "white": np.array([255, 255, 255]),
            "gray": np.array([128, 128, 128]),
            "cyan": np.array([0, 255, 255]),
            "magenta": np.array([255, 0, 255]),
            "lime": np.array([0, 255, 0]),
            "indigo": np.array([75, 0, 130]),
            "violet": np.array([238, 130, 238]),
            "turquoise": np.array([64, 224, 208]),
            "beige": np.array([245, 245, 220]),
            "lavender": np.array([230, 230, 250]),
            "coral": np.array([255, 127, 80]),
            "gold": np.array([255, 215, 0]),
            "silver": np.array([192, 192, 192]),
            "maroon": np.array([128, 0, 0]),
            "navy": np.array([0, 0, 128]),
            "teal": np.array([0, 128, 128]),
            "olive": np.array([128, 128, 0]),
            "salmon": np.array([250, 128, 114]),
            "plum": np.array([221, 160, 221]),
            "chocolate": np.array([210, 105, 30]),
            "tan": np.array([210, 180, 140]),
            "peach": np.array([255, 229, 180]),
            "crimson": np.array([220, 20, 60]),
            "aqua": np.array([0, 255, 255]),
            "ivory": np.array([255, 255, 240]),
            "orchid": np.array([218, 112, 214]),
            "khaki": np.array([240, 230, 140]),
            "mint": np.array([189, 252, 201]),
            "amber": np.array([255, 191, 0]),
            "ruby": np.array([224, 17, 95]),
            "emerald": np.array([80, 200, 120]),
            "jade": np.array([0, 168, 107]),
            "bronze": np.array([205, 127, 50]),
            "sapphire": np.array([15, 82, 186]),
            "periwinkle": np.array([204, 204, 255]),
            "slate": np.array([112, 128, 144]),
            "amethyst": np.array([153, 102, 204]),
            "fuchsia": np.array([255, 0, 255]),
            "azure": np.array([240, 255, 255]),
            "charcoal": np.array([54, 69, 79]),
            "rose": np.array([255, 0, 127])
        }


    def contains(self, rgb):
        """
        Whether the class contains a given rgb code.
        Parameters
        ----------
        rgb: 1D nd.array of size 3

        Returns
        -------
        contains: Bool
            True if rgb code in given Color class.
        """

        return (self.colors_rgb[self.color] == np.array(rgb)).all()

    def sample(self):
        """
        Sample an rgb code from the Color class

        Returns
        -------
        rgb: 1D nd.array of size 3
        """
        return self.colors_rgb[self.color]


def sample_color(color):
    """
    Sample an rgb code from the Color class

    Parameters
    ----------
    color: str
        Color in red, blue, green.
    shade: str
        Shade in light, dark.

    Returns
    -------
    rgb: 1D nd.array of size 3
    """
    color_class = Color(color)
    return color_class.sample()
