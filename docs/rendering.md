# Rendering (`rendering/`)

`PygameRenderer` draws **offscreen by default**. The window is reserved for `present()`; obtaining a framebuffer never opens a window or initialises audio, rendering never changes the simulation, and the same state always produces identical pixels.

```python
from rendering import PygameRenderer, ViewConfig, ViewMode

renderer = PygameRenderer(headless=True)
follow = renderer.render_rgb(game)                          # (160, 240, 3)
renderer.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=None))
full = renderer.render_rgb(game)                            # (H, W, 3) = world size
renderer.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320), downscale=2))
small = renderer.render_rgb(game)                           # (160, 240, 3)
```

## Views

```python
class ViewMode(Enum):
    FOLLOW = "follow"
    FULL = "full"

@dataclass(frozen=True, slots=True)
class ViewConfig:
    mode: ViewMode = ViewMode.FOLLOW
    viewport_size: tuple = (240, 160)       # (w, h); FOLLOW only
    canvas_size: tuple | None = (480, 320)  # (w, h); FULL only; None = world size
    downscale: int = 1                      # integer factor; render_rgb/grayscale only
```

- **`FOLLOW`** (default) shows a 240×160 window centred on the player, clamped to the world edges, through the stateless camera in `game/camera.py` (`compute_viewport`, a pure function with no pygame or NumPy).
- **`FULL`** draws the whole world. With `canvas_size=None` the canvas is the world size, so the frame shape **varies per level**. With `canvas_size=(w, h)` the canvas is fixed: the world is drawn 1:1 anchored top-left and the rest is `VOID` (black), so frame shapes stay constant across levels. The world is never rescaled to fit; if it does not fit, `ValueError` ("use FOLLOW or a larger canvas").
- **`downscale=k`** is an exact integer block average of `k×k`. It requires `k` to divide both dimensions (`ValueError` otherwise) and is ignored by `present()`. `FOLLOW` is byte-for-byte a crop of `FULL`.

Frames are `uint8` with image axes `(height, width, 3)` and do not share memory with the drawing surface.

### Reference configurations

These are the three configurations the RL layer uses:

| Name | `ViewConfig` | `render_rgb` shape |
| --- | --- | --- |
| `follow` | `ViewConfig(FOLLOW)` | `(160, 240, 3)` |
| `full` | `ViewConfig(FULL, canvas_size=(480, 320))` | `(320, 480, 3)` |
| `full_small` | `ViewConfig(FULL, canvas_size=(480, 320), downscale=2)` | `(160, 240, 3)` |

The shape is dictated by the `ViewConfig`, not the level. `follow` and `full_small` share the exact shape `(160, 240, 3)`, so the same network can compare partial and full observability.

## API

```python
class PygameRenderer:
    def __init__(self, scale: int = 3, headless: bool = False, view: ViewConfig | None = None): ...
    def set_view(self, view: ViewConfig) -> None: ...
    def draw(self, game: Game) -> pygame.Surface: ...
    def render_rgb(self, game: Game) -> np.ndarray:          # (H, W, 3) uint8
    def render_grayscale(self, game: Game) -> np.ndarray:    # (H, W) uint8
    def present(self, game: Game) -> None: ...
    def close(self) -> None: ...
```

- `scale` is the **window** scale for `present()` (a positive integer, otherwise `ValueError`), independent of `downscale`, which applies to the framebuffer.
- With `headless=True`, `present()` raises `RuntimeError`.
- `render_rgb` returns an **independent copy**: two consecutive frames never share memory.
- `render_grayscale` uses integer BT.601 luminance, `(299·R + 587·G + 114·B) // 1000`, widened to `uint32` before multiplying to avoid overflow.

## Appearance

Constant colours, defined in `rendering/palette.py` (standard library only, so the [dashboard](dashboard.md) serves the same palette and shape rules to its canvas renderer), no randomness and no clock:

| Colour | RGB | Encodes |
| --- | --- | --- |
| `BACKGROUND` | `(24, 29, 48)` | background (air) |
| `VOID` | `(0, 0, 0)` | outside the canvas/world in fixed `FULL` |
| `PLATFORM` | `(67, 80, 103)` | platform body |
| `PLATFORM_TOP` | `(114, 170, 137)` | platform top edge (2 px) |
| `PLAYER` | `(255, 209, 102)` | living player |
| `PLAYER_DEAD` | `(180, 70, 91)` | dead player |
| `PLAYER_WON` | `(146, 235, 162)` | winning player |
| `HAZARD` | `(239, 91, 91)` | spikes (triangles) |
| `GEM` | `(91, 228, 233)` | gem |
| `EXIT` | `(116, 94, 160)` | **closed** door |
| `EXIT_EDGE` | `(48, 39, 67)` | closed door border |
| `EXIT_INTERIOR` | `(217, 241, 226)` | **open** door interior |

Door semantics, which matter for visual observations:

- **Closed** (no gem): `EXIT` fill, `EXIT_EDGE` border and a small diamond (a gem glyph) in the centre.
- **Unlockable** (gem collected, door not yet touched): the border turns `GEM` (cyan).
- **Open** (`exit_open`): `EXIT_INTERIOR` fill and a `GEM` border.
- **Carried gem**: a small `GEM` diamond drawn at `(player_x + 4, player_y - 4)`, just above the player, while `gem_collected` is `True`.
- If `win_condition == "gem"` and a door is explicitly present, the door is always drawn closed and decorative.

## Frame cost

`render_rgb`, milliseconds per frame, mean of 500 frames (Apple M4 Pro):

| Level | `follow` | `full` | `full_small` |
| --- | --- | --- | --- |
| `level_001` (240×160) | 0.277 | 0.977 | 2.050 |
| `level_007` (480×320) | 0.251 | 0.963 | 2.053 |

`full_small` costs more than `full`: the integer block average over the full frame is itself about 0.5 ms. It reduces the frame's shape and bytes, not the drawing cost. Compared with a pure `step` (a few microseconds), rendering dominates throughput whenever pixels are used.

## Concurrency and caching

- A `PygameRenderer` is **not thread-safe**: it keeps mutable state. Use **one renderer per thread or process**. It is cheap to create because it opens no window.
- The static layer (platforms and spikes) is cached by **`Level` identity**. Switching `game.level` to a new object invalidates the cache and redraws it. This is negligible next to a step, but matters if a new level object is built every episode: keep the `Level` object stable per worker (`LevelDistribution.get(i)` caches it).
- A renderer can be shared by several `Game` instances on the **same `Level`**. Nothing is shared across processes, nor the `pygame.display` unless the process initialises it.
- To batch observations, use `follow` or `full_small`, never `canvas_size=None`, whose frame shape changes with the level (`level_008`, 720 px wide, does not fit a 480×320 canvas).
