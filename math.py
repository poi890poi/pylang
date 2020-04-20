from itertools import chain


def synchronized_intervals(intervals_list):
    '''
    Run a single pointer over a list of intervals.
    '''
    interval_indices = [-1,] * len(intervals_list)
    intervals_current = [None,] * len(intervals_list)

    intervals_all = list(chain.from_iterable(intervals_list))
    intervals_all = sorted(list(set(chain.from_iterable(intervals_all))))

    for i, index in enumerate(interval_indices):
        begin, end = intervals_list[i][0]
        if begin > 0:
            interval_indices = -1 # The list has no interval at this position

    for splitter in intervals_all:
        for i, index in enumerate(interval_indices):
            try:
                index_next = index + 1
                begin, end = intervals_list[i][index_next]
                if splitter >= begin:
                    interval_indices[i] = index_next
                    intervals_current[i] = intervals_list[i][index_next]
                else:
                    try:
                        begin, end = intervals_current[i]
                        if splitter >= end:
                            intervals_current[i] = None
                    except TypeError:
                        pass
            except IndexError:
                pass

        yield interval_indices, intervals_current


if __name__ == '__main__':
    intervals_list = [
        [(0, 3), (3, 7), (7, 9), (9, 14), (14, 19)],
        [(0, 6), (6, 7), (7, 12), (12, 13), (15, 17)],
    ]
    print(intervals_list)
    for interval_indices, intervals_current in synchronized_intervals(intervals_list):
        print(interval_indices, intervals_current)