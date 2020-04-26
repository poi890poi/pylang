import re
import codecs
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
        if 0 <= ord(c) <= 127:
            half_width += 1
        else:
            full_width += 1
    return half_width, full_width


class SplittedComponents(str):
    '''
    Split text by spaces. Return intervals and splitted components.
    '''

    RE_EMPTY = re.compile('^[ ]*$')
    RE_SPACES_SPLITTING = re.compile('([ ]*)([^ ]+)([ ]*)')
    RE_METRICS = re.compile('(\\d+)([^ \\d]+)')
    RE_CODE = re.compile('([A-Z][A-Z\\d]+)')
    RE_DATE = re.compile('(\\d+年)[^\\d]*(\\d+月)')

    def __new__(cls, text, *args, **kwargs):
        return super(SplittedComponents, cls).__new__(cls, text)

    def __init__(self, text, *args, **kwargs):
        self.update_text(text)

    def update_text(self, text):
        self.__text = text
        components = self.RE_SPACES_SPLITTING.findall(text)
        components = [c for c in chain.from_iterable(components) if c]
        self.__components = np.array(components, dtype=str)
        self.__intervals = self.splitters_to_interval([len(c) for c in components])
        self.__empty = np.array([True if self.RE_EMPTY.match(c) else False for c in components], dtype=bool)
        print(self.__components, self.__intervals, self.__empty)

    @classmethod
    def __is_empty(cls, component):
        # TODO A more sophiscated method to include full-wdith space is required
        return component.strip()

    @property
    def text(self):
        return self.__text

    @property
    def intervals(self):
        return self.__intervals

    @property
    def components(self):
        return self.__components

    @property
    def nonempty_splitters(self):
        # Backward compatibility for legacy codes
        splitters = [interval[1] for i, interval in 
            enumerate(self.__intervals) if not self.__empty[i]]
        return splitters

    @property
    def splitters(self):
        # Backward compatibility for legacy codes
        _, splitters = list(zip(*self.__intervals))
        return splitters

    @classmethod
    def splitters_to_interval(cls, splitters):
        end = np.cumsum(splitters)
        begin = np.copy(end)
        begin = np.roll(begin, 1)
        begin[0] = 0
        begin = begin.reshape(-1, 1)
        end = end.reshape(-1, 1)
        return np.hstack((begin, end)).astype(np.uint)

    @property
    def nonempty_components(self):
        components = [c for i, c in enumerate(self.__components) if not self.__empty[i]]
        return components

    @property
    def stats(self):
        # Find number of numeric components in the line
        n_nonempty = 0
        n_numeric = 0
        n_metric = 0
        n_code = 0
        n_date = 0
        for c in self.__components:
            if self.__is_empty(c):
                n_nonempty += 1
            try:
                float(c)
                n_numeric += 1
            except ValueError:
                m = self.RE_METRICS.match(c)
                if m:
                    n_metric += 1
                m = self.RE_CODE.match(c)
                if m:
                    n_code += 1
                m = self.RE_DATE.match(c)
                if m:
                    n_date += 1
        return n_numeric, n_metric, n_code, n_date, n_nonempty

    def pad_spaces(self):
        '''
        Align half-width and full-width chars by adding spaces to the left 
        of components. Assume that full-width is twice the width of half-width
        chars.
        '''
        #residue = 0. # TODO Handle width ratio that can't be divided
        components = self.__components.tolist()
        if self.__text:
            for i, c in enumerate(self.__components):
                if not c.strip():
                    spaces = 0
                    spaces = len(c)
                    try:
                        half, full = count_width_chars(self.__components[i+1])
                        padding = full
                        spaces += padding
                    except IndexError:
                        pass
                    components[i] = ' ' * spaces
            self.update_text(''.join(components))


if __name__ == '__main__':
    scomponents = SplittedComponents('       This is a pen.  ')
    print(scomponents.components)
    print(scomponents.intervals)
    print(scomponents.nonempty_components)
    print(scomponents.splitters)
    print(scomponents.nonempty_splitters_right)