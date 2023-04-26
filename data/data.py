import os
from glob import glob
from skimage.io import imread
from skimage.io import imsave

BASE_DIR = os.getcwd()

DATASETS = {
    'gs_rainfall_daily': 'gs/GPM_3IMERGDL'
    }

def download(
    dataset=DATASETS['gs_rainfall_daily'],
    years=tuple(range(2015, 2021)),
    ext='.PNG',
    force=False,
    verbose='q',
    ) -> str:
    """
    Standardize downloading data

    Parameters
    ----------
    dataset : string
        dataset to download in <type>/<name> format (e.g. rgb or geotiff)
    years : iterable
        years in dataset to download
    ext : string
        file extension (e.g. .JPEG or .TIFF)
    force : bool
        force downloading if dataset already exists
    verbose : str
        verbosity of wget: q (quiet), nv (not verbose), or v (verbose)

    Returns
    -------
    string
         path to saved data
    """

    url = 'https://neo.gsfc.nasa.gov/archive/' + dataset
    save_dir = f'{BASE_DIR}/data/datasets/{dataset}/raw'

    # \ in triple quotes for os.system
    template = """wget --no-directories --no-host-directories --no-parent \
    --recursive --mirror \
    --accept {accept} \
    -l1 {url}/""" \
    + f""" -P {save_dir} \
    -{verbose}"""

    # Skip if data exists
    if os.path.exists(save_dir) and (not force):
        return save_dir

    # Download color table if needed for dataset
    if dataset == DATASETS['gs_rainfall_daily']:
        os.system(
            template.format(
                accept='*.act',
                url='https://neo.gsfc.nasa.gov/palettes/trmm_rainfall.act'
                )
            )

    # Download PNGs
    os.system(
        template.format(
            accept=','.join(map(lambda y: f'*{y}*{ext}', years)),
            url=url
            )
        )

    return save_dir

def process_gs_rainfall_daily(force=False, n_images=-1, log=100) -> str:
    """
    Perform standard processing for gs_rainfall_daily

    Parameters
    ----------
    force : bool
        force processing if processed data already exists
    n_images : int
        number of images to process (used for testing), -1 indicates all
    log : int
        print every log images, -1 means no logging

    Returns
    -------
    string
        path to processed data
    """

    raw_dir = f"{BASE_DIR}/data/datasets/{DATASETS['gs_rainfall_daily']}/raw"
    processed_dir = f"{BASE_DIR}/data/datasets/{DATASETS['gs_rainfall_daily']}/processed"

    # Skip if processed data exists and not reprocessing
    if os.path.exists(processed_dir) and (not force):
        return processed_dir

    # Make processed_dir if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # Get file paths
    raw_files = sorted(glob(raw_dir + '/*.PNG'))
    if n_images != -1:
        raw_files = raw_files[:n_images]

    # Cropping limits
    image_size_raw = imread(raw_files[0]).shape
    north_lim = 300
    south_lim = image_size_raw[0]//2
    east_lim = image_size_raw[1]//2

    for file_n, file in enumerate(raw_files):

        if (log != -1) and (file_n % log == 0):
            print(f'Processing file number {file_n} ({file.split("/")[-1]})')

        # Read, scale pixels to [0.0, 1.0], crop, and save
        image_arr = imread(file) / 255.
        image_arr = image_arr[north_lim : south_lim, :east_lim].astype('uint8')
        processed_file = processed_dir + '/' \
                            + file.split('/')[-1][:-4] + '_processed.PNG'
        imsave(processed_file, image_arr, check_contrast=False)

    return processed_dir
