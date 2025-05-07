"""Defines utilities for overlays in the Mujoco viewer."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Tuple

import mujoco as mj
import numpy as np
import torch

from cambrian.renderer.render_utils import resize_with_aspect_fill
from cambrian.utils.logger import get_logger

TEXT_HEIGHT = 20
TEXT_MARGIN = 5


@dataclass(slots=True)
class MjCambrianCursor:
    class Position(Enum):
        TOP_LEFT = 1
        TOP_RIGHT = 2
        BOTTOM_LEFT = 3
        BOTTOM_RIGHT = 4

    class Layer(Enum):
        BACK = 1
        AUTO = 2
        FRONT = 3

    container_width: int | None = None
    container_height: int | None = None
    position: Position = Position.TOP_LEFT
    layer: Layer = Layer.AUTO
    margin: int = TEXT_MARGIN
    x: int | None = None
    y: int | None = None

    def __post_init__(self):
        if self.container_width is not None and self.container_height is not None:
            self.set_position()
        else:
            assert (
                self.x is not None and self.y is not None
            ), "Either container width and height or x and y must be provided."

    def set_position(self, x: int | None = Ellipsis, y: int | None = Ellipsis):
        if x is not Ellipsis:
            self.x = x
        if y is not Ellipsis:
            self.y = y

        if self.position == MjCambrianCursor.Position.TOP_LEFT:
            x = self.margin
            y = self.container_height - self.margin - TEXT_HEIGHT
        elif self.position == MjCambrianCursor.Position.TOP_RIGHT:
            x = self.container_width - self.margin
            y = self.container_height - self.margin - TEXT_HEIGHT
        elif self.position == MjCambrianCursor.Position.BOTTOM_LEFT:
            x = self.margin
            y = self.margin - TEXT_HEIGHT
        elif self.position == MjCambrianCursor.Position.BOTTOM_RIGHT:
            x = self.container_width - self.margin
            y = self.margin

        if self.x is None:
            self.x = x
        if self.y is None:
            self.y = y

    def move(self, dx: int, dy: int) -> "MjCambrianCursor":
        self.x += dx
        self.y += dy
        return self

    def copy(self) -> "MjCambrianCursor":
        return replace(self)


class MjCambrianViewerOverlay:
    """This class is used to add an overlay to the viewer.

    Note:
        This is applied only to the passed scene, so other scenes (i.e. ones for the
        eyes) will not be affected.
    """

    def __init__(self, obj: Any, cursor: Optional[MjCambrianCursor] = None):
        self._obj = obj
        self._cursor = cursor.copy() if cursor is not None else None

    def draw_before_render(self, scene: mj.MjvScene):
        """Called before rendering the scene."""
        pass

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        """Called after rendering the scene."""
        pass

    def place(self, cursor: MjCambrianCursor) -> MjCambrianCursor:
        """Places the cursor at the given location. Won't overwrite the current cursor,
        if it exists.

        Args:
            cursor: The cursor to place.
        """
        if self._cursor is None:
            self._cursor = cursor.copy()
        return cursor

    @property
    def layer(self) -> MjCambrianCursor.Layer:
        return (
            self._cursor.layer
            if self._cursor is not None
            else MjCambrianCursor.Layer.AUTO
        )

    # =============

    @staticmethod
    def create_text_overlay(
        text: str,
        *,
        cursor: MjCambrianCursor | None = None,
    ):
        """Creates a text overlay."""
        return MjCambrianTextViewerOverlay(text, cursor)

    @staticmethod
    def create_image_overlay(
        obj: torch.Tensor, *, cursor: MjCambrianCursor | None = None
    ):
        """Creates an image overlay."""
        return MjCambrianImageViewerOverlay(obj, cursor)

    @staticmethod
    def create_site_overlay(
        pos: np.ndarray,
        rgba: Tuple[float, float, float, float],
        size: float,
        geom_kwargs: dict = dict(emission=0.25),
    ):
        """Creates a site overlay."""
        return MjCambrianSiteViewerOverlay(pos, rgba, size, geom_kwargs)


class MjCambrianTextViewerOverlay(MjCambrianViewerOverlay):
    """This class is used to add text to the viewer."""

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        viewport = (
            viewport
            if self._cursor is None
            else mj.MjrRect(self._cursor.x, self._cursor.y, 1, 1)
        )
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            self._obj,
            "",
            mjr_context,
        )

    def place(self, cursor: MjCambrianCursor) -> MjCambrianCursor:
        if cursor.position in {
            MjCambrianCursor.Position.BOTTOM_LEFT,
            MjCambrianCursor.Position.BOTTOM_RIGHT,
        }:
            cursor.move(0, TEXT_HEIGHT + TEXT_MARGIN)
        else:
            cursor.move(0, -(TEXT_HEIGHT + TEXT_MARGIN))

        return super().place(cursor)


class MjCambrianImageViewerOverlay(MjCambrianViewerOverlay):
    """This class is used to add an image to the viewer."""

    def __init__(self, obj: torch.Tensor, cursor: Optional[MjCambrianCursor] = None):
        super().__init__(obj, cursor)
        self._obj_cpu = obj.cpu()

    def draw_after_render(self, mjr_context: mj.MjrContext, viewport: mj.MjrRect):
        assert self._cursor is not None
        self._obj_cpu.copy_(self._obj, non_blocking=True)
        viewport = mj.MjrRect(
            self._cursor.x - self._cursor.margin,
            self._cursor.y - self._cursor.margin,
            self._obj.shape[1],
            self._obj.shape[0],
        )
        mj.mjr_drawPixels(self._obj_cpu.numpy().ravel(), None, viewport, mjr_context)

    def place(self, cursor: MjCambrianCursor) -> MjCambrianCursor:
        # Resize the image to fit the container
        self._obj = resize_with_aspect_fill(
            self._obj, cursor.container_height, cursor.container_width
        )
        self._obj_cpu = self._obj.cpu()

        # Adjust the cursor position
        self._cursor.x = cursor.x

        return super().place(cursor)


class MjCambrianSiteViewerOverlay(MjCambrianViewerOverlay):
    """This class is used to add a site to the viewer.

    Todo:
        Make this an image overlay where the pos is converted to pixel coordinates.
    """

    def __init__(
        self,
        pos: np.ndarray,
        rgba: Tuple[float, float, float, float],
        size: float,
        geom_kwargs: dict = dict(emission=0.25),
    ):
        super().__init__(pos)
        self._rgba = rgba
        self._size = size
        self._geom_kwargs = geom_kwargs

    def draw_before_render(self, scene: mj.MjvScene):
        if scene.ngeom >= scene.maxgeom:
            get_logger().warning(
                f"Max geom reached ({scene.maxgeom}). Cannot add more sites."
            )
            return

        scene.ngeom += 1
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mj.mjtGeom.mjGEOM_SPHERE,
            [self._size] * 3,
            self._obj,
            np.eye(3).flatten(),
            self._rgba,
        )
        for key, value in self._geom_kwargs.items():
            setattr(scene.geoms[scene.ngeom - 1], key, value)
