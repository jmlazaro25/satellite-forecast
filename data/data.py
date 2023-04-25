import os

BASE_DIR = os.getcwd()

DATASETS = {
    'gs_rainfall_daily': 'gs/GPM_3IMERGDL/'
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
    -l1 {url}""" \
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


def process_gs_rainfall_daily(force=False, n_images=-1):
    """
    Perform standard processing for gs_rainfall_daily

    Parameters
    ----------
    force : bool
        force processing if processed data already exists
    n_images : int
        number of images to process (used for testing), -1 indicates all

    Returns
    -------
    string
        path to processed data
    """

    processed_dir = f"{BASE_DIR}/data/datasets/{DATASETS['gs_rainfall_daily']}/processed"

    # Skip if processed data exists and not reprocessing
    if os.path.exists(processed_dir) and (not force):
        return save_dir

    # Make processed_dir if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)


