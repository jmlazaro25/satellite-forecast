import os
from skimage.io import imread, imsave
from skimage.transform import rescale, resize

from satforecast.data import data

IMAGE_DIR = os.path.join(data.BASE_DIR, 'images')

def get_border_map(force=False):

    """
    Extract borders for plotting on top of weather predictions
    Currently crops to first version of processed images
    Note that we can use `extent` when plotting rather than scaling this to
        match the scaling of the weather data
    """

    raw_map_file = os.path.join(IMAGE_DIR, 'BlankMap-World-Subdivisions.png')
    borders_file = os.path.splitext(raw_map_file)[0] + f'_crop.png'

    if os.path.exists(borders_file) and (not force):
        return borders_file

    # Remove background -> (alpha = 0)
    # Emperical findings:
    #   1. Land is (254, 254, 254, 255)
    #   2. Water and international borders are (255, 255, 255, 255)
    #   3. Intranational borders are (196, 196, 196, 255)
    #   4. Antarctic claims include (129,,,), (239,,,), and land
    # Future consideration: edge detection to keep international borders
    land = 254
    water_or_inter = 255
    intra = 196

    raw_map = imread(raw_map_file)
    water_or_inter_mask = (raw_map[:,:,0] == water_or_inter)
    intra_mask = (raw_map[:,:,0] == intra)

    # Use edge detection to find costlines and international borders
    borders = raw_map.copy()
    borders[~water_or_inter_mask, 0] = 0 # Temporarily remove other features
    from skimage.feature import canny
    coast_or_inter = canny(borders[:,:,0], sigma=1)

    # Set interborder, intraborder, and other rgba
    borders[coast_or_inter] = (0, 0, 0, 127) # Black, alpha = 50%
    borders[intra_mask] = (0, 0, 0, 63) # Black, alpha = 25%
    borders[~intra_mask & ~coast_or_inter, 3] = 0

    # Raw weather image for size reference
    raw_dir = data.download()
    raw_image_file = data.get_files(raw_dir, '*.PNG')[0]
    image_size_raw = imread(raw_image_file).shape

    # Reshape map so coordinates ~~ line up with the weather images
    borders = resize(
        borders,
        (image_size_raw[0], image_size_raw[1]),
        anti_aliasing=True,
        preserve_range=True
    )

    # Crop as in data.process_gs_rainfall_daily for alignment
    north_lim = 300
    south_lim = image_size_raw[0]//2
    east_lim = image_size_raw[1]//2
    borders = borders[north_lim : south_lim, :east_lim]

    # Save
    imsave(borders_file, borders.astype('uint8'))

    return borders_file
