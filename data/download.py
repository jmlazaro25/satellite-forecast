import os

BASE_DIR = os.getcwd()

DATASETS = {
    'gs_rainfall_daily': 'gs/GPM_3IMERGDL/'
    }

def download(
    dataset=DATASETS['gs_rainfall_daily'],
    years=list(range(2015, 2021)),
    ext='.PNG',
    force=False,
    verbose='q',
    ):
    """ Standardize downloading data """

    url = 'https://neo.gsfc.nasa.gov/archive/' + dataset
    save_dir = f'{BASE_DIR}/data/{dataset}'

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
