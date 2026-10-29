"""Explicit USB-camera diagnostic; importing this module never opens a device."""

import argparse
import time


def list_cameras(cv2, count=10):
    available = []
    for index in range(count):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened() and capture.read()[0]:
                available.append(index)
        finally:
            capture.release()
    return available


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Probe indices 0 through 9 and exit')
    parser.add_argument('--device', type=int, default=0, help='Camera index to preview (default: 0)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.fps) <= 0 or args.device < 0:
        parser.error('Dimensions and FPS must be positive; device index must be nonnegative')

    import cv2

    if args.list:
        cameras = list_cameras(cv2)
        print('Available camera indices:', cameras)
        return 0 if cameras else 1
    capture = cv2.VideoCapture(args.device)
    try:
        if not capture.isOpened():
            print(f'Could not open camera {args.device}')
            return 1
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        print(f'Camera {args.device}; reported FPS: {capture.get(cv2.CAP_PROP_FPS)}; press q to close')
        while True:
            started = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                print(f'Could not read camera {args.device}')
                return 1
            cv2.imshow(f'Camera {args.device}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return 0
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
