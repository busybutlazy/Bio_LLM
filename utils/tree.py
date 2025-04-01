from pathlib import Path
import os
import argparse

# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '


def tree(dir_path: Path, prefix: str='',layer=300):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """    
    counter=layer
    if counter<=0:
        return
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension,layer=layer-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print directory tree")
    parser.add_argument(
        "--path", 
        type=str, 
        default=".", 
        help="Root path to generate tree from (default: current directory)"
    )
    parser.add_argument(
        "--depth", 
        type=int, 
        default=2, 
        help="Depth of tree (default: 2)"
    )
    args = parser.parse_args()

    root_path = Path(args.path).resolve()
    depth = args.depth

    print(root_path.name)
    for line in tree(root_path, '', layer=depth):
        print(line)