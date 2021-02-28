#!/usr/bin/env python3
# simple script to convert seconds to ffmpeg hh:mm:ss format
import sys

def to_base(n: int, b: int) -> list:
    """ converts a number from base 10 to base b. """
    l = []
    while n != 0:
        l.append(n % b)
        n //= b
    return l[::-1]

def format_time(t: int) -> str:
    """ Formats a time in seconds to hh:mm:ss. """
    digit = to_base(t, 60)
    return ":".join(str(d).rjust(2, "0") for d in [0]*(3 - len(digit)) + digit)

def format_ffmpeg(start: int, end: int) -> str:
    """ Returns a ffmpeg start and end time.
    Note: -t is a duration and -to is an ending timestamp. """
    return f"-ss {format_time(start)} -to {format_time(end)}"

if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    print(format_ffmpeg(start, end))

