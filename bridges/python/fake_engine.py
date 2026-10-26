"""A tiny "native game" that speaks the GlitchHunter bridge protocol, for tests and as a reference.

It renders a 2D scene with OpenCV: a floor, a crate that slides while the D key is held, and
an optional planted glitch (`--glitch missing_texture` paints the crate magenta/black). It is
the smallest complete implementation of the protocol in docs/ENGINES.md:

    python fake_engine.py --gh-port 5555 [--width 320 --height 180] [--glitch missing_texture]

`--unseeded` adds film grain from an unseeded generator, the classic way an engine integration
breaks determinism (bridge-check catches it).
"""

from __future__ import annotations

import argparse
import base64
import json
import socket

import cv2 as cv
import numpy as np


class Game:
    def __init__(self, width: int, height: int, glitch: str | None, unseeded: bool = False):
        self.w, self.h, self.glitch = width, height, glitch
        self.grain = np.random.default_rng() if unseeded else None
        self.reset(1)

    def reset(self, seed: int) -> None:
        self.frame = 0
        self.crate_x = 40.0
        self.keys: set[str] = set()
        self.camera = {"pos": [0.0, 0.0, 10.0], "target": [0.0, 0.0, 0.0], "fov": 55.0}

    def advance(self, n: int) -> None:
        for _ in range(n):
            self.frame += 1
            if "KeyD" in self.keys:
                self.crate_x += 4.0

    def crate_box(self) -> tuple[int, int, int, int]:
        shift = int(self.camera["pos"][0] * 10)
        x = int(self.crate_x) - shift
        return x, self.h // 2 - 20, 48, 48

    def render(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), (110, 120, 130), np.uint8)
        img[self.h // 2 + 28:] = (70, 80, 90)
        x, y, w, h = self.crate_box()
        if self.glitch == "missing_texture":
            for j in range(0, h, 8):
                for i in range(0, w, 8):
                    color = (255, 0, 255) if (i + j) // 8 % 2 == 0 else (0, 0, 0)
                    cv.rectangle(img, (x + i, y + j), (x + i + 7, y + j + 7), color, -1)
        else:
            cv.rectangle(img, (x, y), (x + w, y + h), (40, 90, 150), -1)
            cv.line(img, (x, y), (x + w, y + h), (30, 60, 100), 3)
        if self.grain is not None:
            img = cv.add(img, self.grain.integers(0, 6, img.shape, dtype=np.uint8))
        return img

    def pick(self, px: float, py: float) -> dict:
        x, y, w, h = self.crate_box()
        if x <= px <= x + w and y <= py <= y + h:
            return {"hit": True, "object": "crate", "point": [px / 10, py / 10, 0.0], "distance": 10.0,
                    "coplanar": ["crate"], "camera": self.camera}
        return {"hit": True, "object": "floor", "point": [px / 10, py / 10, -1.0], "distance": 11.0,
                "coplanar": ["floor"], "camera": self.camera}


def serve(port: int, game: Game) -> None:
    conn = socket.create_connection(("127.0.0.1", port))
    stream = conn.makefile("rwb")

    def send(obj: dict) -> None:
        stream.write((json.dumps({"ok": True, **obj}) + "\n").encode("utf-8"))
        stream.flush()

    for line in stream:
        msg = json.loads(line)
        cmd = msg.get("cmd")
        if cmd == "hello":
            send({"api": 1, "engine": "fake-engine 1.0", "viewport": [game.w, game.h]})
        elif cmd == "reset":
            game.reset(int(msg.get("seed", 1)))
            send({})
        elif cmd == "set_camera":
            game.camera.update({k: msg[k] for k in ("pos", "target", "fov") if k in msg})
            send({"camera": game.camera})
        elif cmd == "key":
            (game.keys.add if msg.get("down", True) else game.keys.discard)(msg["code"])
            send({})
        elif cmd == "advance":
            game.advance(int(msg.get("frames", 1)))
            send({"frame": game.frame})
        elif cmd == "grab":
            ok, buf = cv.imencode(".png", game.render())
            send({"png": base64.b64encode(buf.tobytes()).decode("ascii"), "size": [game.w, game.h]})
        elif cmd == "state":
            send({"frame": game.frame, "alive": True, "errors": [], "camera": game.camera})
        elif cmd == "pick":
            send(game.pick(float(msg["x"]), float(msg["y"])))
        elif cmd == "quit":
            send({})
            break
        else:
            stream.write((json.dumps({"ok": False, "error": f"unknown command {cmd}"}) + "\n").encode())
            stream.flush()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gh-port", type=int, required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--glitch", default=None)
    parser.add_argument("--unseeded", action="store_true")
    args = parser.parse_args()
    serve(args.gh_port, Game(args.width, args.height, args.glitch, args.unseeded))


if __name__ == "__main__":
    main()
