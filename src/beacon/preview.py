"""Optional preview window. Shows ONLY the blurred frame plus trust overlay.

DeepFace depends on the *headless* OpenCV build (no cv2.imshow), so the window
uses Tkinter + Pillow, which ship with the python.org Windows installer.
"""

from __future__ import annotations

import numpy as np

from .privacy import BlurredFrame, require_blurred
from .trust import Decision, State

_COLORS = {State.TRUSTED: "#2e7d32", State.SUSPECT: "#f9a825", State.LOCK_REQUIRED: "#c62828"}


class PreviewUnavailable(RuntimeError):
    pass


class TkPreview:
    def __init__(self, title: str = "BEACON (blurred preview)", max_width: int = 640) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise PreviewUnavailable(
                "preview needs Tkinter and Pillow (Tkinter ships with the python.org installer)"
            ) from exc
        self._tk, self._Image, self._ImageTk = tk, Image, ImageTk
        self.max_width = max_width
        self.stop_requested = False
        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._stop)
        self.root.bind("<Key-q>", lambda e: self._stop())
        self.label = tk.Label(self.root)
        self.label.pack()
        self.bar = tk.Canvas(self.root, height=26, bg="#222")
        self.bar.pack(fill="x")
        self.text = tk.Label(self.root, font=("Consolas", 11), anchor="w", justify="left")
        self.text.pack(fill="x")
        self._photo = None

    def _stop(self) -> None:
        self.stop_requested = True

    def show(self, blurred: BlurredFrame | None, decision: Decision, dry_run: bool) -> None:
        if blurred is not None:
            img = require_blurred(blurred).image
            h, w = img.shape[:2]
            scale = min(1.0, self.max_width / w)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            pil = self._Image.fromarray(rgb)
            if scale < 1.0:
                pil = pil.resize((int(w * scale), int(h * scale)))
            self._photo = self._ImageTk.PhotoImage(pil)
            self.label.configure(image=self._photo)
        width = max(1, self.bar.winfo_width())
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, int(width * decision.trust), 26, fill=_COLORS[decision.state], width=0)
        mode = "DRY-RUN (no real lock)" if dry_run else "REAL LOCK ENABLED"
        self.text.configure(
            text=f"{decision.state.value}  trust={decision.trust:.2f}  {decision.reason}\n{mode}   press q to quit"
        )
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
