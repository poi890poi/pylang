import re
from itertools import chain

import numpy as np


def count_half_chars(text):
    count = 0
    for c in text:
        if 0 <= ord(c) <= 127:
            count += 1
    return count

def count_full_chars(text):
    count = 0
    for c in text:
        if not (0 <= ord(c) <= 127):
            count += 1
    return count

def count_width_chars(text):
    half_width = 0
    full_width = 0
    for c in text:
        if not (0 <= ord(c) <= 127):
            half_width += 1
        else:
            full_width += 1
    return half_width, full_width


class SplittedComponents(str):

    __RE_EMPTY = re.compile('^[ ]*$')
    __RE_SPACES_SPLITTING = re.compile('([ ]*)([^ ]+)([ ]*)')

    def __new__(cls, text, *args, **kwargs):
        return super(SplittedComponents, cls).__new__(cls, text)

    def __init__(self, text, *args, **kwargs):
        self.__text = text
        components = self.__RE_SPACES_SPLITTING.findall(text)
        components = np.array([c for c in chain.from_iterable(components) if c])
        self.__components = components
        end = np.cumsum([len(c) for c in components])
        begin = np.hstack(([0,], end[:-1])).reshape(-1, 1)
        end = end.reshape(-1, 1)
        self.__intervals = np.hstack((begin, end)).astype(np.uint)
        self.__empty = np.array([True if self.__RE_EMPTY.match(c) else False for c in components], dtype=bool)
        print(self.__components, self.__intervals, self.__empty)

    @property
    def nonempty_components(self):
        components = [c for i, c in enumerate(self.__components) if not self.__empty[i]]
        return components


def splitted_components(text):
    '''
    Split text by spaces. Return intervals and splitted components.
    '''
    pass

if __name__ == '__main__':
    line = SplittedComponents('       This is a pen.  ')
    print(line.nonempty_components)