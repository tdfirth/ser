from typing import List

import typer

from ser.transforms import flip

TRANSFORM_DICT = {"flip": flip}


def transform_callback(transform_ls: List[str]):
    ls_set = set(transform_ls)

    transform_func_ls = []
    for transform in ls_set:
        if transform not in TRANSFORM_DICT.keys():
            raise typer.BadParameter(
                f"{transform} not in available list of transforms: {','.join([str(x) for x in TRANSFORM_DICT.keys()])}"
            )
        transform_func_ls.append(TRANSFORM_DICT[transform])
    return transform_func_ls
