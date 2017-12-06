TRAINING_DATA_PATH = 'E://mydata/'


def get_directory_name():
    return '%sIMG/' % TRAINING_DATA_PATH


def get_file_name(path, is_windows=True):
    separator = '\\' if is_windows else '/'
    return path.split(separator)[-1]


def get_path(directory_name, file_name):
    return '%s%s' % (directory_name, file_name)


def convert_path(source):
    return get_directory_name() + get_file_name(source)


def greyscale(X):
    return X[:, :, :, :1] / 3 + X[:, :, :, 1:2] / 3 + X[:, :, :, -1:] / 3


def reversing(X):
    return X[:, :, :, ::-1]


def greyscale_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[-1] = 1
    return tuple(shape)


def read_logs(csv_file_name):
    with open(get_path(TRAINING_DATA_PATH, csv_file_name)) as csvfile:
        lines = [l.strip('\n').split(',') for l in csvfile.readlines()]
    return lines
