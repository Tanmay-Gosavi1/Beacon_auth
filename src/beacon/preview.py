"""Optional preview window. Shows ONLY the blurred frame plus trust overlay.

DeepFace depends on the *headless* OpenCV build (no cv2.imshow), so the window
uses Tkinter + Pillow, which ship with the python.org Windows installer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import numpy as np

from .liveness import Liveness
from .privacy import BlurredFrame, require_blurred
from .trust import Decision, State

if TYPE_CHECKING:
    from .face_engine import FaceBox
    from .pipeline import FrameResult, Pipeline

_COLORS = {State.TRUSTED: "#2e7d32", State.SUSPECT: "#f9a825", State.LOCK_REQUIRED: "#c62828"}


class PreviewUnavailable(RuntimeError):
    pass


def check_console_ready() -> str | None:
    """Non-blocking keyboard check for console Enter / Space / q on Windows."""
    try:
        import msvcrt

        while msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b"\r", b"\n", b" "):
                return "ready"
            if ch in (b"q", b"Q", b"\x1b"):
                return "stop"
    except Exception:
        pass
    return None


def check_positioning_frame(r: FrameResult) -> str | None:
    """Check positioning frame suitability (ignoring missing embedding)."""
    if r.face_count == 0:
        return "no face detected"
    if r.face_count > 1:
        return f"more than one face detected ({r.face_count})"
    if r.identity_reason == "face_too_small":
        return "face too small (move closer)"
    if r.liveness is not None and r.liveness.label is not Liveness.LIVE:
        reason = r.liveness.reason if r.liveness else "uncertain"
        return f"liveness: {reason}"
    return None


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

    def show(
        self,
        blurred: BlurredFrame | None,
        decision: Decision,
        dry_run: bool,
        similarity: float | None = None,
    ) -> None:
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
        sim_text = f"  sim={similarity:.2f}" if similarity is not None else ""
        self.text.configure(
            text=f"{decision.state.value}  trust={decision.trust:.2f}{sim_text}  {decision.reason}\n{mode}   press q to quit"
        )
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


class TkEnrollmentPreview:
    """Enrollment preview window showing privacy-blurred camera frames,

    pose instructions, bounding boxes, live rejection reasons, and progress.
    """

    def __init__(self, title: str = "BEACON - Guided Face Enrollment", max_width: int = 640) -> None:
        try:
            import tkinter as tk
            from PIL import Image, ImageDraw, ImageTk
        except ImportError as exc:
            raise PreviewUnavailable(
                "preview needs Tkinter and Pillow (Tkinter ships with the python.org installer)"
            ) from exc
        self._tk, self._Image, self._ImageDraw, self._ImageTk = tk, Image, ImageDraw, ImageTk
        self.max_width = max_width
        self.stop_requested = False
        self._ready = False

        self.root = tk.Tk()
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self._stop)
        self.root.bind("<Key-q>", lambda e: self._stop())
        self.root.bind("<Escape>", lambda e: self._stop())
        self.root.bind("<Return>", lambda e: self._on_ready())
        self.root.bind("<KP_Enter>", lambda e: self._on_ready())
        self.root.bind("<space>", lambda e: self._on_ready())

        # Header for current pose
        self.pose_header = tk.Label(
            self.root,
            font=("Segoe UI", 12, "bold"),
            bg="#222222",
            fg="#ffffff",
            pady=8,
            anchor="center",
            text="Initializing enrollment...",
        )
        self.pose_header.pack(fill="x")

        # Camera frame label
        self.label = tk.Label(self.root, bg="#000000")
        self.label.pack()

        # Visual progress bar (Canvas)
        self.bar = tk.Canvas(self.root, height=18, bg="#2a2a2a", highlightthickness=0)
        self.bar.pack(fill="x")

        # Status & feedback label
        self.text = tk.Label(
            self.root,
            font=("Consolas", 10),
            bg="#222222",
            fg="#e0e0e0",
            anchor="w",
            justify="left",
            padx=10,
            pady=6,
        )
        self.text.pack(fill="x")

        # Control panel with Ready button and cancel hint
        ctrl_frame = tk.Frame(self.root, bg="#1a1a1a", pady=6)
        ctrl_frame.pack(fill="x")

        self.btn = tk.Button(
            ctrl_frame,
            text="Ready / Capture (Space or Enter)",
            font=("Segoe UI", 10, "bold"),
            bg="#2e7d32",
            fg="#ffffff",
            activebackground="#1b5e20",
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=5,
            cursor="hand2",
            command=self._on_ready,
        )
        self.btn.pack(side="left", padx=10)

        hint = tk.Label(
            ctrl_frame,
            text="Press 'q' or Esc to cancel",
            font=("Segoe UI", 9),
            bg="#1a1a1a",
            fg="#888888",
        )
        hint.pack(side="right", padx=10)

        self._photo = None

    def _on_ready(self) -> None:
        self._ready = True

    def _stop(self) -> None:
        self.stop_requested = True
        self._ready = True

    def show_frame(
        self,
        blurred: BlurredFrame | None,
        boxes: tuple[FaceBox, ...] = (),
        status_text: str = "",
        status_color: str = "#e0e0e0",
        progress: float = 0.0,
    ) -> None:
        if self.stop_requested:
            return
        if blurred is not None:
            img = require_blurred(blurred).image
            h, w = img.shape[:2]
            scale = min(1.0, self.max_width / w)
            rgb = np.ascontiguousarray(img[:, :, ::-1])
            pil = self._Image.fromarray(rgb)
            if scale < 1.0:
                pil = pil.resize((int(w * scale), int(h * scale)))

            if boxes:
                draw = self._ImageDraw.Draw(pil)
                for b in boxes:
                    x0 = int(b.x * scale)
                    y0 = int(b.y * scale)
                    x1 = int((b.x + b.w) * scale)
                    y1 = int((b.y + b.h) * scale)
                    draw.rectangle([x0, y0, x1, y1], outline=status_color, width=3)

            self._photo = self._ImageTk.PhotoImage(pil)
            self.label.configure(image=self._photo)

        width = max(1, self.bar.winfo_width())
        self.bar.delete("all")
        self.bar.create_rectangle(
            0, 0, int(width * max(0.0, min(1.0, progress))), 18, fill="#2e7d32", width=0
        )
        self.text.configure(text=status_text, fg=status_color)
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            self.stop_requested = True

    def wait_for_ready(
        self,
        pose_index: int,
        total_poses: int,
        pose_desc: str,
        read_frame: Callable[[], np.ndarray],
        pipeline: Pipeline,
        pause_s: float = 0.03,
    ) -> None:
        import threading
        from .privacy import blur_with

        self._ready = False
        self.pose_header.configure(text=f"Pose [{pose_index}/{total_poses}]: {pose_desc}")
        self.btn.configure(state="normal", text="Ready / Capture (Space or Enter)", bg="#2e7d32")

        worker_lock = threading.Lock()
        latest_boxes: tuple[FaceBox, ...] = ()
        latest_status: str = "Position yourself in front of the camera, then press Space/Enter or click Ready"
        latest_box_color: str = "#f9a825"
        worker_busy = False
        worker_thread: threading.Thread | None = None

        def _worker_task(f: np.ndarray) -> None:
            nonlocal latest_boxes, latest_status, latest_box_color, worker_busy
            try:
                r = pipeline.process(f, want_embedding=False)
                err = check_positioning_frame(r) if r else "pipeline error"
                with worker_lock:
                    latest_boxes = r.boxes if r else ()
                    if err is None:
                        latest_box_color = "#2e7d32"
                        latest_status = "Status: Face in position! Press Space/Enter or click Ready"
                    else:
                        latest_box_color = "#f9a825"
                        latest_status = f"Status: {err}\nPosition yourself, then press Space/Enter or click Ready"
            except Exception:
                with worker_lock:
                    latest_boxes = ()
                    latest_box_color = "#f9a825"
                    latest_status = "Status: Positioning check...\nPosition yourself, then press Space/Enter or click Ready"
            finally:
                del f
                with worker_lock:
                    worker_busy = False

        while not self._ready and not self.stop_requested:
            cin = check_console_ready()
            if cin == "ready":
                self._ready = True
                break
            if cin == "stop":
                self.stop_requested = True
                break

            try:
                frame = read_frame()
            except Exception:
                break

            try:
                blurred = blur_with(frame, pipeline.privacy)
            except Exception:
                blurred = None

            with worker_lock:
                should_spawn = (not worker_busy) and (not self._ready) and (not self.stop_requested)
                if should_spawn:
                    worker_busy = True

            if should_spawn:
                worker_thread = threading.Thread(target=_worker_task, args=(frame.copy(),), daemon=True)
                worker_thread.start()

            del frame

            with worker_lock:
                boxes = latest_boxes
                status = latest_status
                box_color = latest_box_color

            progress = (pose_index - 1) / total_poses
            self.show_frame(
                blurred,
                boxes=boxes,
                status_text=status,
                status_color=box_color,
                progress=progress,
            )
            time.sleep(pause_s)

        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=3.0)

        if not self.stop_requested:
            self.btn.configure(state="disabled", text="Capturing samples...", bg="#555555")
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception:
                self.stop_requested = True

    def show_sample(
        self,
        r: FrameResult | None,
        pose_index: int,
        total_poses: int,
        pose_desc: str,
        got: int,
        needed: int,
        attempts: int,
        max_attempts: int,
        rejection: str | None,
    ) -> None:
        if self.stop_requested:
            return
        cin = check_console_ready()
        if cin == "stop":
            self.stop_requested = True
            return

        overall_progress = ((pose_index - 1) + (got / needed)) / total_poses
        if rejection is None:
            color = "#2e7d32"
            status = f"Sample accepted ({got}/{needed}) [Attempt {attempts}/{max_attempts}]"
        else:
            color = "#c62828"
            status = f"Sample rejected: {rejection} ({got}/{needed} accepted) [Attempt {attempts}/{max_attempts}]"

        self.show_frame(
            r.blurred if r else None,
            boxes=r.boxes if r else (),
            status_text=status,
            status_color=color,
            progress=overall_progress,
        )

    def show_completed(self, total_samples: int) -> None:
        if self.stop_requested:
            return
        self.pose_header.configure(text="Enrollment Complete!", fg="#2e7d32")
        self.btn.configure(state="disabled", text="Finished", bg="#555555")
        self.show_frame(
            None,
            boxes=(),
            status_text=f"Successfully captured {total_samples} samples across all poses.",
            status_color="#2e7d32",
            progress=1.0,
        )

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass

