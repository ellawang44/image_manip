from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional, TypeVar

# Fallback for Protocol import
try:
    from typing import Protocol  # type: ignore
except Exception:  # pragma: no cover
    from typing_extensions import Protocol  # type: ignore

import numpy as np

# Fallback for NDArray type hinting
if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore
else:
    NDArray = Any  # type: ignore
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

###############################################################################
### Type Definitions ##########################################################
###############################################################################

T = TypeVar("T", covariant=True)
U = TypeVar("U", bound=np.generic, contravariant=True)
V = TypeVar("V", bound=np.generic, covariant=True)


@dataclass
class QuantizedColourImage:
    """Dataclass for quantized colour image."""

    image_data: NDArray[np.int_]
    centers: NDArray[np.float32]


PixelFormat = Literal["single", "cycle"]

###############################################################################
### Protocols #################################################################
###############################################################################


class ImageTemplate(Protocol[T, U]):
    def __call__(self, x: NDArray[U]) -> T: ...


class CharFormater(Protocol[V]):
    def __call__(self, i: int, centers: NDArray[np.float32]) -> V: ...


###############################################################################
### Image Template Functions ##################################################
###############################################################################


def str_template(x: NDArray[np.str_]) -> np.str_:
    """String template function."""
    final_str = ""
    for row in x:
        final_str += "".join(row) + "\n"
    return np.str_(final_str[:-1])


def html_str_template(x: NDArray[np.str_], font_size: int = 14) -> np.str_:
    """HTML string template function."""
    str_vals = str_template(x)
    final_html = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <pre style="font-size: {font_size}px; line-height: 1ch;">{str_vals}</pre>
    </body>
    </html>
    """
    return np.str_(final_html)


###############################################################################
### Image Manipulation Class ##################################################
###############################################################################


class ImageManipulate:
    """Manipulate images."""

    def __init__(self, path_to_image: str) -> None:
        """
        Parameters
        ----------
        path_to_image : str
            Path to the image to be converted.
        """

        self.path_to_image = path_to_image
        self._load_img_data()

    def _load_img_data(self) -> None:
        """Load image data from path."""

        image = Image.open(self.path_to_image)
        image = ImageOps.exif_transpose(image)

        self.image = image.convert("RGB")

    def resize(self, hresize: Optional[int], vresize: Optional[int]) -> None:
        """Resize image.

        Parameters
        ----------
        hresize : Optional[int]
            Horizontal size to resize to. None retains original ratio.
        vresize : Optional[int]
            Vertical size to resize to. None retains original ratio.
        """
        horig, vorig = self.image.size
        if hresize is None and vresize is not None:
            scale = vresize / vorig
            hresize = int(horig * scale)
            resize = (hresize, vresize)

        elif vresize is None and hresize is not None:
            scale = hresize / horig
            vresize = int(vorig * scale)
            resize = (hresize, vresize)

        elif hresize is not None and vresize is not None:
            resize = (hresize, vresize)

        else:
            return

        self.image = self.image.resize(resize)

    def get_colour_quantize_image(
        self, components: int = 20, seed: Optional[int] = None
    ) -> QuantizedColourImage:
        """Quantize the image colour with k means clustering.

        Parameters
        ----------
        components : int
            Number of components/characters to convert image into.
        seed : Optional[int]
            The random seed to use for KMeans clustering. Can be None.

        Returns
        -------
        quantized_image : QuantizedColourImage
            The quantized colour image.
        """

        # open and reshape in matplotlib
        image_data = np.array(self.image, dtype=float)

        vpx, hpx, rgb = image_data.shape
        image_data = image_data.reshape(vpx * hpx, rgb)

        # normalise
        scaler = StandardScaler().fit(image_data)
        img_scale = scaler.transform(image_data)

        # fit and apply k means clustering
        kmeans = KMeans(n_clusters=components, random_state=seed)
        kmeans.fit(img_scale)
        # this seems to keep ordering correct as opposed to predict
        # important for cartoon filter, not so much ascii art
        image_quantized = kmeans.labels_
        image_quantized = image_quantized.reshape(vpx, hpx)

        return QuantizedColourImage(
            image_data=image_quantized,
            centers=scaler.inverse_transform(kmeans.cluster_centers_),
        )


###############################################################################
### Quantize Output Functions #################################################
###############################################################################


def quantized_to_formatted_output(
    quantized_image: QuantizedColourImage,
    image_template: ImageTemplate[T, U],
    pixel_formater: CharFormater[U],
) -> T:
    """Convert quantized image to formatted string representation.

    Parameters
    ----------
    quantized_image : QuantizedColourImage
        The quantized colour image.
    str_template : function
        Function to format the final string. If None, default function is used.
    char_formater : function
        Function to format each character. If None, default function is used.

    Returns
    -------
    formatted_str : str
        The image converted to formatted string representation.
    """
    pixel_output = []
    for row in quantized_image.image_data:
        output_row = []
        for v in row:
            pixel = pixel_formater(v, quantized_image.centers)
            output_row.append(pixel)
        pixel_output.append(output_row)
    pixel_output = np.array(pixel_output)
    return image_template(pixel_output)


def quantized_to_ascii_str(
    quantized_image: QuantizedColourImage,
    ascii_str: str = "-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm",
    colour: bool = True,
) -> np.str_:
    """Convert quantized image to ascii art.

    Parameters
    ----------
    quantized_image : QuantizedColourImage
        The quantized colour image.
    ascii_str : str
        The ascii characters to use. If string is longer than components, rest
        of string is ignored.
    colour : bool
        Whether to output ascii art in colour.

    Returns
    -------
    ascii_art : str
        The image converted to ascii art.
    """

    if len(ascii_str) < len(quantized_image.centers):
        raise ValueError(
            "ascii_str is not long enough for the number of k-means components specified."
        )

    def ascii_char_formater(i: int, centers: np.ndarray) -> np.str_:
        char = ascii_str[i]
        if colour:
            r, g, b = centers[i].astype(int)
            char = f"\033[38;2;{r};{g};{b}m" + char + "\033[0m"
        return np.str_(char)

    return quantized_to_formatted_output(
        quantized_image=quantized_image,
        image_template=str_template,
        pixel_formater=ascii_char_formater,
    )


def quantized_to_ascii_html(
    quantized_image: QuantizedColourImage,
    path_to_html: str,
    font_size: int = 14,
    ascii_str: str = "-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm",
    colour: bool = True,
) -> None:
    """Convert quantized image to ascii art in html format.

    Parameters
    ----------
    quantized_image : QuantizedColourImage
        The quantized colour image.
    path_to_html : str
        Path to output html file.
    font_size : int
        Font size to use in html file.
    ascii_str : str
        The ascii characters to use. If string is longer than components, rest
        of string is ignored.
    colour : bool
        Whether to output ascii art in colour.
    """

    if len(ascii_str) < len(quantized_image.centers):
        raise ValueError(
            "ascii_str is not long enough for the number of k-means components specified."
        )

    def html_ascii_char_formater(i: int, centers: np.ndarray) -> np.str_:
        char = ascii_str[i]
        if colour:
            r, g, b = centers[i].astype(int)
            hex_str = f"#{r:02x}{g:02x}{b:02x}"
            char = f'<span style="color:{hex_str}">{char}</span>'
        return np.str_(char)

    html_template = functools.partial(html_str_template, font_size=font_size)
    html_str = quantized_to_formatted_output(
        quantized_image=quantized_image,
        image_template=html_template,
        pixel_formater=html_ascii_char_formater,
    )

    # Export to html file
    if not path_to_html.endswith(".html"):
        path_to_html += ".html"

    with open(path_to_html, "w") as f:
        f.write(html_str)


def quantized_to_cartoon_file(
    quantized_image: QuantizedColourImage,
    path_to_output: str,
) -> None:
    """Convert quantized image to cartoon filtered image.

    Parameters
    ----------
    quantized_image : QuantizedColourImage
        The quantized colour image.

    Returns
    -------
    cartoon_image : NDArray[np.float32]
        The cartoon filtered image.
    """

    def cartoon_char_formater(i: int, centers: np.ndarray) -> np.str_:
        return centers[i]

    cartoon_image = quantized_to_formatted_output(
        quantized_image=quantized_image,
        image_template=lambda x: x,
        pixel_formater=cartoon_char_formater,
    )

    # Convert float image in [0,1] to uint8 and save with PIL
    out_img = cartoon_image.astype(np.uint8)
    Image.fromarray(out_img).save(path_to_output)


def _pixel_generator(
    pixel_chars: str,
    pixel_format: PixelFormat,
) -> Iterator[str]:
    """Generate pixel characters based on format.

    Parameters
    ----------
    pixel_chars : str
        Character(s) for pixel art.
    pixel_format : PixelFormat
        Pixel format to use.

    Returns
    -------
    pixel_gen : Iterator[str]
        Iterator for pixel characters.
    """
    if pixel_format == "single":
        if len(pixel_chars) != 1:
            raise ValueError("pixel_char is invalid, must be str of len 1.")

        pixel_gen = itertools.repeat(pixel_chars)
    elif pixel_format == "cycle":
        pixel_gen = itertools.cycle(pixel_chars)
    else:
        raise ValueError("pixel_format is invalid.")

    return pixel_gen


def quantized_to_pixelart_str(
    quantized_image: QuantizedColourImage,
    pixel_chars: str = "█",
    pixel_format: PixelFormat = "single",
) -> np.str_:
    """Convert quantized image to pixel art.

    Parameters
    ----------
    quantized_image : QuantizedColourImage
        The quantized colour image.
    pixel_chars: str
        Character(s) for pixel art.
    pixel_format : PixelFormat
        Pixel format to use.

    Returns
    -------
    pixel_art_image : NDArray[np.uint8]
        The pixel art image.
    """
    pixel_gen = _pixel_generator(pixel_chars, pixel_format)

    def pixel_char_formater(i: int, centers: np.ndarray) -> np.str_:
        pixel_char = next(pixel_gen)
        r, g, b = centers[i].astype(int)
        return np.str_(f"\033[38;2;{r};{g};{b}m{pixel_char}\033[0m")

    return quantized_to_formatted_output(
        quantized_image=quantized_image,
        image_template=str_template,
        pixel_formater=pixel_char_formater,
    )


def quantized_to_pixelart_html(
    quantized_image: QuantizedColourImage,
    path_to_html: str,
    pixel_chars: str = "█",
    pixel_format: PixelFormat = "single",
    font_size: int = 14,
) -> None:
    """Convert quantized image to pixel art.

    Parameters
    ----------
    quantized_image : QuantizedColourImage
        The quantized colour image.
    path_to_html : str
        Path to output html file.
    pixel_chars: str
        Character(s) for pixel art.
    pixel_format : PixelFormat
        Pixel format to use.
    font_size : int
        Font size to use in html file.
    """
    pixel_gen = _pixel_generator(pixel_chars, pixel_format)

    def pixel_char_formater(i: int, centers: np.ndarray) -> np.str_:
        pixel_char = next(pixel_gen)
        r, g, b = centers[i].astype(int)
        hex_str = f"#{r:02x}{g:02x}{b:02x}"
        return np.str_(
            f'<span style="background-color: {hex_str}; color:{hex_str}">{pixel_char}</span>'
        )

    html_template = functools.partial(html_str_template, font_size=font_size)
    pixel_output = quantized_to_formatted_output(
        quantized_image=quantized_image,
        image_template=html_template,
        pixel_formater=pixel_char_formater,
    )

    # Export to html file
    if not path_to_html.endswith(".html"):
        path_to_html += ".html"

    with open(path_to_html, "w") as f:
        f.write(pixel_output)
