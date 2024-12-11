def lower_clamp(v):
    return 0 if v < 0 else v


def upper_clamp(v, max_v):
    return max_v if v > max_v else v
