"""The window: resizable, maximize, and F11 fullscreen, drawn with an accelerated renderer.

Everything is drawn on `surface` (the 3D view plus panels) at a capped render resolution,
then uploaded once per frame as a texture that the GPU scales to the window, so a large or
maximized window costs no more to draw than the cap. Presenting does not wait for the screen's
refresh (VSYNC below); the app paces frames on its own 60 Hz schedule. On the tested Windows
desktop (Direct3D 11, windowed) the compositor showed whole frames; other platforms, renderers,
or exclusive fullscreen may behave differently.
"""

from __future__ import annotations

import math

import pygame

MIN_SIZE = (960, 540)
MAX_RENDER_PIXELS = 1920 * 1080  # the render resolution is scaled down beyond this
PREFERRED_DRIVERS = ("direct3d11", "direct3d12", "opengl", "metal")
# The app paces frames itself (robot_env/app.py present_deadline). A vsync wait is not used: on
# Windows a vsync present of a covered or background window can block for about 250 ms (measured),
# which freezes the simulation. Whole frames (no tearing) were observed in the tested Windows
# Direct3D 11 window; fullscreen and other systems may differ.
VSYNC = False
MAX_RENDER_SIZE = (2560, 1440)  # MuJoCo's offscreen buffer (world.xml <global offwidth/offheight>)


def render_size(window: tuple[int, int]) -> tuple[int, int]:
    """Drawing resolution for a window size: the window size, scaled down (same aspect ratio)
    to at most MAX_RENDER_PIXELS and MAX_RENDER_SIZE, never below MIN_SIZE's height."""
    w, h = max(window[0], MIN_SIZE[0]), max(window[1], MIN_SIZE[1])
    s = min(1.0, math.sqrt(MAX_RENDER_PIXELS / (w * h)), MAX_RENDER_SIZE[0] / w, MAX_RENDER_SIZE[1] / h)
    return max(int(w * s) // 2 * 2, 2), max(int(h * s) // 2 * 2, 2)


class Display:
    def __init__(self, size: tuple[int, int], title: str):
        from pygame._sdl2 import video
        self._video = video
        self.window = video.Window(title, size=size, resizable=True)
        # Direct3D 11 first: SDL's default Direct3D 9 renderer crashes once MuJoCo's OpenGL
        # renderer has been recreated in the same process (found in testing).
        names = [d.name for d in video.get_drivers()]
        index = next((names.index(n) for n in PREFERRED_DRIVERS if n in names), -1)
        self.renderer = video.Renderer(self.window, index=index, vsync=VSYNC, accelerated=1)
        self.driver = names[index] if index >= 0 else "default"
        self.fullscreen = False
        self.surface = pygame.Surface(render_size(size))
        self._texture = None
        self._texture_size = None

    @property
    def window_size(self) -> tuple[int, int]:
        return tuple(self.window.size)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Window events. True when the drawing size changed (the caller rebuilds its renderer)."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
            self.toggle_fullscreen()
        if event.type in (pygame.WINDOWSIZECHANGED, pygame.WINDOWRESIZED, pygame.WINDOWMAXIMIZED,
                          pygame.WINDOWRESTORED) or (event.type == pygame.KEYDOWN and event.key == pygame.K_F11):
            return self.sync_size()
        return False

    def sync_size(self) -> bool:
        w, h = self.window_size
        if not self.fullscreen and (w < MIN_SIZE[0] or h < MIN_SIZE[1]):
            self.window.size = (max(w, MIN_SIZE[0]), max(h, MIN_SIZE[1]))  # pygame 2.6 has no minimum-size hint
        size = render_size(self.window_size)
        if size == self.surface.get_size():
            return False
        self.surface = pygame.Surface(size)
        return True

    def toggle_fullscreen(self) -> None:
        if self.fullscreen:
            self.window.set_windowed()
        else:
            self.window.set_fullscreen(desktop=True)
        self.fullscreen = not self.fullscreen

    def present(self) -> None:
        """Upload the frame and hand it to the renderer (does not wait for the refresh)."""
        size = self.surface.get_size()
        if self._texture is None or self._texture_size != size:
            self._texture = self._video.Texture(self.renderer, size, streaming=True)
            self._texture_size = size
        self._texture.update(self.surface)
        self.renderer.clear()
        self._texture.draw(dstrect=(0, 0, *self.window_size))
        self.renderer.present()

    def close(self) -> None:
        self._texture = None
        self.renderer = None
        self.window.destroy()
