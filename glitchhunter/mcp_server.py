"""MCP server: the OpenCV 5 game-QA tools for any MCP client (Claude Code, Kiro, ...).

Run:  python -m glitchhunter mcp                      (stdio)
      python -m glitchhunter mcp --transport streamable-http
"""

from __future__ import annotations

from mcp.server import MCPServer

from . import service

INSTRUCTIONS = (
    "GlitchHunter tools: capture a build of the game headlessly (scenarios come from the repo's "
    "glitchhunter.json), run OpenCV 5 glitch detectors on the frames, ray-cast into the scene, compare to "
    "the approved golden build, and bisect commits with a detector as the test. Typical flow: "
    "capture -> detect -> (pick / capture closer / other screen size) -> bisect -> show_commit."
)


def build_server() -> MCPServer:
    server = MCPServer(name="glitchhunter", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool()
    def opencv_info() -> dict:
        """OpenCV build facts (version, DNN engines, CPU features), the scenarios and the detectors."""
        return service.info()

    @server.tool()
    def list_commits(limit: int = 30) -> list[dict]:
        """Commits of the game repo, newest first (first-parent history of main)."""
        return service.list_commits(limit)

    @server.tool()
    def show_commit(sha: str) -> dict:
        """Subject, files and diff of one commit."""
        return service.show_commit(sha)

    @server.tool()
    def capture(commit: str = "main", scenario: str = "overview", camera: dict | None = None, seed: int = 1,
                viewport: list[int] | None = None) -> dict:
        """Play a scenario on a build in headless Chromium and keep the frames. Returns a run_id.

        camera: optional {"pos": [x,y,z], "target": [x,y,z], "fov": deg} to look from somewhere else.
        viewport: optional [width, height] screen size (e.g. [480, 360] for 4:3).
        """
        return service.capture(commit, scenario, camera, seed, viewport)

    @server.tool()
    def detect(run_id: str, detectors: list[str] | None = None, region: list[int] | None = None) -> dict:
        """Run OpenCV 5 glitch detectors on a captured run (optionally only inside region [x,y,w,h])."""
        return service.detect(run_id, detectors, region)

    @server.tool()
    def pick(run_id: str, x: float, y: float, frame: int = 0) -> dict:
        """Ray-cast through pixel (x, y) of a run: which object is there, its 3D point, coplanar surfaces."""
        return service.pick(run_id, x, y, frame)

    @server.tool()
    def compare_to_golden(run_id: str) -> dict:
        """Compare a run with the same view on the approved baseline build: structure (SSIM), tone shift,
        shadow acne, texture detail loss, missing objects, exploded geometry."""
        return service.compare_to_golden(run_id)

    @server.tool()
    def bisect(detector: str, scenario: str, good: str | None = None, bad: str = "main",
               camera: dict | None = None, region: list[int] | None = None, repeats: int = 1,
               viewport: list[int] | None = None) -> dict:
        """Find the first commit where the detector fires (visual git bisect).

        good defaults to the approved baseline. Builds that do not start are skipped; if the first bad
        mainline commit is a merge, the merged branch is searched too.
        """
        return service.bisect(detector, scenario, good, bad, camera, region, repeats, viewport)

    return server
