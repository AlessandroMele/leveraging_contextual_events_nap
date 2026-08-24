from os.path import join, dirname
from os import getcwd, listdir, cpu_count
import warnings
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

DIR_PATH = dirname(join(getcwd(), __file__))
INPUT_PATH = join(DIR_PATH, 'dataset')
LOG_NAME = reversed(sorted([item.split('_processed.g')[0] for item in listdir(INPUT_PATH)
                   if item.endswith('_processed.g')]))

VARIANTS = ['var_1', 'var_2', 'var_3', 'var_4']
MAX_WORKERS = cpu_count()

EPOCHS = 150
N_TRIALS = 20
EARLY_STOPPING = 20

