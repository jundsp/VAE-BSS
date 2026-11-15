"""Animate the training figures of a run (reconstruction_<epoch>.png) into a GIF.

    python -m src.animate results/audio_synthetic_c0.5_vae_K2 -o docs/imgs/audio_training.gif

Early epochs change the most, so every figure up to ``--dense-until`` is used and only every
``--step``-th epoch after it. Each frame is resized to ``--width`` pixels, labelled with its
epoch above the figure, and the last frame is held for ``--hold`` seconds.
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def figures(folder):
    """``(epoch, path)`` of every reconstruction figure in ``folder``, by epoch."""
    found = []
    for path in Path(folder).glob('reconstruction_*.png'):
        match = re.fullmatch(r'reconstruction_(\d+)\.png', path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def subsample(found, dense_until, step):
    """Every figure up to ``dense_until``, then every ``step`` epochs, always the last."""
    keep = [(e, p) for e, p in found if e <= dense_until or e % step == 0]
    if found and keep[-1] != found[-1]:
        keep.append(found[-1])
    return keep


def label(image, text):
    """The figure under a white strip with ``text`` (the epoch), so no panel is covered."""
    size = max(14, image.width // 45)
    try:
        font = ImageFont.load_default(size=size)  # Pillow >= 10.1: a scalable built-in font
    except TypeError:
        font = ImageFont.load_default()
    box = ImageDraw.Draw(image).textbbox((0, 0), text, font=font)
    band = box[3] - box[1] + 14
    framed = Image.new('RGB', (image.width, image.height + band), 'white')
    framed.paste(image, (0, band))
    ImageDraw.Draw(framed).text((10, 5), text, fill='black', font=font)
    return framed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run', help='Run folder with reconstruction_<epoch>.png figures')
    parser.add_argument('-o', '--out', default=None,
                        help='Output GIF (default: <run>/training.gif)')
    parser.add_argument('--width', type=int, default=1000, help='Frame width in pixels')
    parser.add_argument('--dense-until', type=int, default=100,
                        help='Use every figure up to this epoch (default: 100)')
    parser.add_argument('--step', type=int, default=30,
                        help='After that, one figure every this many epochs (default: 30)')
    parser.add_argument('--fps', type=float, default=4, help='Frames per second')
    parser.add_argument('--hold', type=float, default=2.5,
                        help='Seconds to hold the last frame')
    args = parser.parse_args(argv)

    found = figures(args.run)
    if not found:
        raise SystemExit(f'no reconstruction_<epoch>.png figures in {args.run}')
    frames = []
    for epoch, path in subsample(found, args.dense_until, args.step):
        image = Image.open(path).convert('RGB')
        height = round(image.height * args.width / image.width)
        image = image.resize((args.width, height), Image.LANCZOS)
        frames.append(label(image, f'epoch {epoch}'))

    out = Path(args.out or Path(args.run) / 'training.gif')
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [round(1000 / args.fps)] * len(frames)
    durations[-1] = round(1000 * args.hold)
    # One shared palette keeps the colours steady between frames and the file small.
    palette = frames[-1].quantize(colors=128, method=Image.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    print(f'{len(frames)} frames (epochs {found[0][0]}-{found[-1][0]}) -> {out} '
          f'({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
