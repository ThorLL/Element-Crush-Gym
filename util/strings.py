def add_padding(string, padding, left=0, right=0, spacing=1):
    return f"{padding * left}{' ' * spacing}{string}{' ' * spacing}{padding * right}"


def align_left(string, length, padding, spacing=1):
    return add_padding(string, padding, left=length - len(string), spacing=spacing)


def align_right(string, length, padding, spacing=1):
    return add_padding(string, padding, right=length - len(string), spacing=spacing)


def align_center(string, length, padding, spacing=1, overflow_right=True):
    missing = length - len(string)
    half_missing = int(missing / 2)
    if overflow_right:
        left, right = half_missing, missing - half_missing
    else:
        left, right = missing - half_missing, half_missing

    return add_padding(string, padding, left, right, spacing)
