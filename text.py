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
