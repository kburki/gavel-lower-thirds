#!/usr/bin/env python3
"""
Bar Locator for Gavel Alaska lower thirds

Finds the lower-third graphic by its colors instead of fixed coordinates,
then OCRs each part separately:

    +--------------------------------------------------------------+
    | NAVY bar:  line 1 (name / bill / location)                   |
    |            line 2 (title, affiliation)        -- optional    |
    +--------------------------------------+-----------------------+
    | TEAL bar:  committee / floor session | KTOO | date  | time   |  <- grey info boxes
    +--------------------------------------+-----------------------+

The teal bar is the anchor: it is always present while the graphic is up.
The navy bar is found by scanning upward from the teal bar, so one-line and
two-line titles both work, and a teal-only graphic (no navy) is handled too.

Usage (standalone test):
    python3 bar_locator.py test_images/josephson.jpg
    python3 bar_locator.py test_images/*.jpg --debug-dir debug_frames
    python3 bar_locator.py --video video/sample5.mp4 --start 2:00 --end 2:30 -v

Video mode skips frames caught mid-dissolve: it compares the graphic area to a
frame ~0.2s earlier and only reads text once the graphic has settled.
"""

import os
os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # Tesseract is ~100x slower multithreaded

import argparse
import re
from collections import Counter
from difflib import SequenceMatcher
import cv2
import numpy as np
import pytesseract

# --- Colors measured from real Gavel footage (BGR order, as OpenCV uses) ---
NAVY_BGR = (73, 40, 20)     # RGB (20, 40, 73)
TEAL_BGR = (131, 100, 7)    # RGB (7, 100, 131)
COLOR_TOLERANCE = 40        # max per-channel distance to count as a match

# A row counts as part of a bar when at least this fraction of its pixels match.
TEAL_ROW_FRACTION = 0.30    # teal spans ~55% of the frame width
NAVY_ROW_FRACTION = 0.35    # measured within the bar's own width; white text lowers it
MIN_TEAL_HEIGHT_PX = 15     # teal bar is ~40px tall on a 720p frame


def _color_mask(img, bgr, tol=COLOR_TOLERANCE):
    """Boolean mask of pixels within `tol` of a BGR color on every channel."""
    diff = np.abs(img.astype(np.int16) - np.array(bgr, dtype=np.int16))
    return np.all(diff <= tol, axis=2)


def _longest_run(flags):
    """Return (start, end_exclusive) of the longest run of True values, or None."""
    best, start = None, None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if best is None or (i - start) > (best[1] - best[0]):
                best = (start, i)
            start = None
    return best


def locate_bars(frame):
    """
    Find the navy bar, teal bar and grey info boxes in a frame.

    Works on a full 1280x720 frame or on a cropped strip of the bottom.

    Returns a dict of boxes as (x1, y1, x2, y2), or None if no teal bar is found:
        {'teal': ..., 'info': ... or None, 'navy': ... or None}
    """
    h, w = frame.shape[:2]

    # Only search the bottom half of a full frame (a strip is searched whole).
    y_off = h // 2 if h > 400 else 0
    region = frame[y_off:]

    # 1. Teal bar: the longest run of rows with enough teal pixels.
    teal = _color_mask(region, TEAL_BGR)
    teal_rows = teal.mean(axis=1) >= TEAL_ROW_FRACTION
    run = _longest_run(teal_rows)
    if run is None or (run[1] - run[0]) < MIN_TEAL_HEIGHT_PX:
        return None
    ty1, ty2 = run

    # Horizontal extent of teal (columns that are mostly teal within those rows).
    teal_cols = teal[ty1:ty2].mean(axis=0) >= 0.5
    cols = np.where(teal_cols)[0]
    tx1, tx2 = int(cols.min()), int(cols.max()) + 1

    # 2. Info boxes: light, low-saturation pixels to the right of the teal bar.
    info = None
    strip = region[ty1:ty2, tx2:]
    if strip.size:
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        light = ((hsv[..., 2] > 140) & (hsv[..., 1] < 60)).mean(axis=0)
        # Smooth so dark text strokes inside the boxes don't break the run,
        # then take the run of light columns that starts right at the teal edge.
        light = np.convolve(light, np.ones(15) / 15, mode='same')
        flags = light >= 0.4
        if flags[:5].any():
            end = int(np.argmin(flags[5:])) + 5 if not flags[5:].all() else len(flags)
            info = (tx2, ty1, tx2 + end, ty2)

    # The navy bar spans the full graphic width: from teal's left edge to the
    # right edge of the info boxes (or teal's right edge if no boxes found).
    gx1 = tx1
    gx2 = info[2] if info else tx2

    # 3. Navy bar: scan upward from the teal bar. The bar's left margin (before
    #    the text starts) is solid navy on every row, which cleanly separates it
    #    from dark background above, even background that is navy-ish overall.
    margin = _color_mask(region[:, gx1 + 4:gx1 + 20], NAVY_BGR).mean(axis=1)
    navy_frac = _color_mask(region[:, gx1:gx2], NAVY_BGR).mean(axis=1)
    ny2 = ty1
    ny1 = ty1
    y = ty1 - 1
    misses = 0
    while y >= 0:
        if margin[y] >= 0.8 and navy_frac[y] >= NAVY_ROW_FRACTION:
            ny1 = y
            misses = 0
        else:
            misses += 1
            if misses > 3:
                break
        y -= 1
    navy_box = (gx1, ny1, gx2, ny2) if (ny2 - ny1) >= 20 else None

    def shift(b):
        return None if b is None else (b[0], b[1] + y_off, b[2], b[3] + y_off)

    return {'teal': shift((tx1, ty1, tx2, ty2)), 'info': shift(info), 'navy': shift(navy_box)}


def _text_lines(crop):
    """
    Split a bar crop into text lines.

    First find each line's core (rows with plenty of bright text), then extend
    each core up and down to catch thin parts like the tails of g/p/y, which
    otherwise get cut off (g then reads as q or a). A line never extends past
    the halfway point to its neighbor, so lines that nearly touch stay separate.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lit = (gray > 160).sum(axis=1)
    core = lit > crop.shape[1] * 0.01
    runs, start = [], None
    for i, c in enumerate(list(core) + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append([start, i])
            start = None
    if not runs:
        return []
    tallest = max(b - a for a, b in runs)
    runs = [r for r in runs if (r[1] - r[0]) >= tallest * 0.5]   # drop slivers
    h = crop.shape[0]
    out = []
    for k, (a, b) in enumerate(runs):
        lo = 0 if k == 0 else (runs[k - 1][1] + a) // 2
        hi = h if k == len(runs) - 1 else (b + runs[k + 1][0]) // 2
        while a > lo and lit[a - 1] >= 2:
            a -= 1
        while b < hi and lit[b] >= 2:
            b += 1
        out.append((max(a - 2, lo), min(b + 2, hi)))
    return out


def _clean(text):
    """Trim stray edge marks OCR picks up at bar edges, and fix bracket misreads."""
    text = text.replace('[', '(').replace(']', ')').replace('{', '(').replace('}', ')')
    text = re.sub(r'^[\s|=»«—_~\'"`.,:;]+', '', text)
    text = re.sub(r'[\s|=»«—_~\'"`,:;-]+$', '', text)
    return text.strip()


def _prep(crop):
    """Simple preprocessing for clean broadcast text: gray, 2x, Otsu, dark-on-white."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).mean() > 0.5:      # mostly dark background -> make text dark
        bw = cv2.bitwise_not(bw)
    return cv2.copyMakeBorder(bw, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def _ocr_line(img):
    return _clean(pytesseract.image_to_string(img, config='--oem 1 --psm 7'))


def parse_info(text):
    """Pull station, date and time out of the info-box OCR, ignoring stray marks."""
    out = {}
    m = re.search(r'\b(K[A-Z0-9]{3})', text.upper())
    if m:
        out['station'] = m.group(1).replace('0', 'O')
    m = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', text)
    if m:
        out['date'] = m.group(1)
    m = re.search(r'(\d{1,2}:\d{2})\.?\s*([ap])\.?\s*m', text, re.IGNORECASE)
    if m:
        out['time'] = f'{m.group(1)} {m.group(2).lower()}m'
    return out


def read_lower_third(frame, debug_dir=None, name='frame'):
    """
    Locate the bars and OCR each part.

    Returns None if no lower third is present, otherwise:
        {'title_lines': [...],      # navy bar, one entry per line
         'committee': str,          # teal bar text
         'info': str,               # e.g. "KTOO Thu. 5/15/25 9:30 am"
         'boxes': {...}}            # pixel boxes, for debugging
    """
    boxes = locate_bars(frame)
    if boxes is None:
        return None

    pad = 4  # stay inside the bar edges
    result = {'title_lines': [], 'committee': '', 'info': '', 'boxes': boxes}

    if boxes['navy']:
        x1, y1, x2, y2 = boxes['navy']
        navy = frame[y1:y2, x1 + pad:x2 - pad]
        for i, (a, b) in enumerate(_text_lines(navy)):
            img = _prep(navy[a:b])
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f'{name}_navy_line{i + 1}.png'), img)
            text = _ocr_line(img)
            if text:
                result['title_lines'].append(text)

    x1, y1, x2, y2 = boxes['teal']
    teal_img = _prep(frame[y1 + pad:y2 - pad, x1 + pad:x2 - pad])
    result['committee'] = _ocr_line(teal_img)

    info_img = None
    if boxes['info']:
        x1, y1, x2, y2 = boxes['info']
        info_img = _prep(frame[y1 + pad:y2 - pad, x1 + pad:x2 - pad])
        result['info'] = _ocr_line(info_img)
        result.update(parse_info(result['info']))

    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f'{name}_teal.png'), teal_img)
        if info_img is not None:
            cv2.imwrite(os.path.join(debug_dir, f'{name}_info.png'), info_img)
        annotated = frame.copy()
        for key, color in (('navy', (0, 0, 255)), ('teal', (0, 255, 0)), ('info', (255, 0, 255))):
            b = boxes[key]
            if b:
                cv2.rectangle(annotated, (b[0], b[1]), (b[2] - 1, b[3] - 1), color, 2)
        cv2.imwrite(os.path.join(debug_dir, f'{name}_boxes.jpg'), annotated)

    return result


STABLE_MAX_DIFF = 4.0   # mean gray-level change allowed for a "settled" graphic


def graphic_box(boxes):
    """Bounding box around everything that was found (navy + teal + info)."""
    found = [b for b in boxes.values() if b]
    return (min(b[0] for b in found), min(b[1] for b in found),
            max(b[2] for b in found), max(b[3] for b in found))


def is_settled(frame, earlier_frame, boxes, max_diff=STABLE_MAX_DIFF):
    """
    True if the graphic area looks the same as in a slightly earlier frame.

    Lower thirds are opaque and static once they've settled, so the bar pixels
    barely change between frames (just compression noise). During a dissolve
    or an animate-on they change a lot. Returns (settled, diff).
    """
    x1, y1, x2, y2 = graphic_box(boxes)
    a = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(earlier_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = float(np.abs(a - b).mean())
    return diff <= max_diff, diff


def _similar(a, b, threshold=0.85):
    """True if two readings are probably the same text with OCR wobble."""
    return SequenceMatcher(None, a, b).ratio() >= threshold


def _parse_time(t):
    """'00:02:15', '2:15' or '135' -> seconds."""
    parts = [float(p) for p in str(t).split(':')]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _fmt(secs):
    return f'{int(secs // 3600)}:{int(secs % 3600 // 60):02d}:{secs % 60:05.2f}'


def scan_video(path, start, end, step, lookback, verbose):
    """
    Sample a stretch of video and print each settled reading when it changes.

    Reads the video straight through (no seeking), so each frame is decoded
    once. Frames are only converted to images when needed: at each sample
    point and `lookback` seconds either side of it. OCR only runs when the
    graphic looks different from the last one read.
    """
    import sys, time
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f'Could not open {path}', flush=True)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    back = max(1, int(round(lookback * fps)))
    every = max(1, int(round(step * fps)))
    first = int(start * fps)
    last_frame = min(int(end * fps), total - 1) if end else total - 1

    if first > back:                       # jump close to the start once
        cap.set(cv2.CAP_PROP_POS_FRAMES, first - back)
        idx = first - back
    else:
        idx = 0

    wanted = {}          # frame index -> image, for frames we still need
    seg = None           # current segment: start, end, committee, Counter of title variants
    cache = None         # (gray graphic crop, box, result) of last OCR'd graphic
    skipped = ocr_calls = 0
    t0 = time.time()
    next_report = 60.0

    def emit(t_end):
        """Print the current segment, using its most common spelling."""
        nonlocal seg
        if seg:
            title = seg['titles'].most_common(1)[0][0]
            print(f"{_fmt(seg['start'])} - {_fmt(t_end)}  {title}   [{seg['committee']}]",
                  flush=True)
            seg = None

    def evaluate(s):
        nonlocal seg, cache, skipped, ocr_calls
        frame = wanted.get(s)
        earlier, later = wanted.get(s - back), wanted.get(s + back)
        t = s / fps
        if frame is None:
            return
        boxes = locate_bars(frame)
        if boxes is None:
            emit(t)
            return
        settled, diff = True, 0.0
        for other in (earlier, later):
            if other is not None and settled:
                settled, diff = is_settled(frame, other, boxes)
        if not settled:
            skipped += 1
            if verbose:
                print(f'{_fmt(t)}  skipped: graphic changing (diff {diff:.1f})', flush=True)
            return
        # Reuse the last OCR result if the graphic looks the same as last time.
        box = graphic_box(boxes)
        x1, y1, x2, y2 = box
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.int16)
        if cache and cache[1] == box and float(np.abs(gray - cache[0]).mean()) <= STABLE_MAX_DIFF:
            r = cache[2]
        else:
            r = read_lower_third(frame)
            ocr_calls += 1
            cache = (gray, box, r)
        title = ' / '.join(r['title_lines']) or '(no title)'
        if seg and _similar(title, seg['titles'].most_common(1)[0][0]) \
                and _similar(r['committee'], seg['committee']):
            seg['titles'][title] += 1          # same graphic, maybe an OCR wobble
            seg['end'] = t
        else:
            emit(t)
            seg = {'start': t, 'end': t, 'committee': r['committee'],
                   'titles': Counter([title])}

    while idx <= last_frame + back:
        if not cap.grab():
            break
        rel = idx - first
        # Is this frame a sample point, or lookback frames either side of one?
        if idx >= 0 and any((rel - o) % every == 0 and first <= idx - o <= last_frame
                            for o in (0, back, -back)):
            ok, img = cap.retrieve()
            if ok:
                wanted[idx] = img
        # Once we've passed sample s + back, everything for sample s is in hand.
        s = idx - back
        if s >= first and (s - first) % every == 0 and s <= last_frame:
            evaluate(s)
        for k in [k for k in wanted if k < idx - 2 * back]:
            del wanted[k]
        idx += 1
        if idx / fps >= next_report:
            print(f'   ... {_fmt(idx / fps)} of {_fmt(total / fps)} '
                  f'({time.time() - t0:.0f}s elapsed)', file=sys.stderr, flush=True)
            next_report += 60.0
    # Samples near the very end that had no "later" frame
    for s in sorted(wanted):
        if s >= first and (s - first) % every == 0 and s > idx - back - 1 and s <= last_frame:
            evaluate(s)
    emit(idx / fps)
    cap.release()
    print(f'-- done in {time.time() - t0:.0f}s; {ocr_calls} OCR call(s); '
          f'skipped {skipped} mid-transition sample(s)', flush=True)


def main():
    p = argparse.ArgumentParser(description='Locate and read Gavel lower-third bars')
    p.add_argument('images', nargs='*', help='Frame image(s) to test')
    p.add_argument('--debug-dir', help='Save annotated frames and OCR inputs here')
    p.add_argument('--video', help='Scan a video instead of still images')
    p.add_argument('--start', default='0', help='Scan start time, e.g. 2:00')
    p.add_argument('--end', default=None, help='Scan end time, e.g. 2:30 (default: end of video)')
    p.add_argument('--step', type=float, default=0.5, help='Seconds between samples (default 0.5)')
    p.add_argument('--lookback', type=float, default=0.2,
                   help='Compare each sample to the frame this many seconds earlier (default 0.2)')
    p.add_argument('-v', '--verbose', action='store_true', help='Also print skipped samples')
    args = p.parse_args()

    if args.video:
        scan_video(args.video, _parse_time(args.start),
                   _parse_time(args.end) if args.end else None,
                   args.step, args.lookback, args.verbose)
        return
    if not args.images:
        p.error('give image file(s), or --video')

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    for path in args.images:
        frame = cv2.imread(path)
        if frame is None:
            print(f'{path}: could not read image')
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        r = read_lower_third(frame, args.debug_dir, name)
        print(f'== {path}')
        if r is None:
            print('   no lower third found')
            continue
        for i, line in enumerate(r['title_lines'], 1):
            print(f'   title {i}:  {line}')
        print(f'   committee: {r["committee"]}')
        print(f'   info:      {r["info"]}')
        print(f'   parsed:    station={r.get("station")}  date={r.get("date")}  time={r.get("time")}')
        print(f'   boxes:     {r["boxes"]}')


if __name__ == '__main__':
    main()
