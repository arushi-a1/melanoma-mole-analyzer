"""
Tkinter desktop GUI for the mole analyzer.

Fixes vs. the original prototype:
  - The original's "control image" concept (comparing the mole to a patch
    of nearby skin) is gone: it's not part of the clinical ABCDE rule and
    added a confusing, buggy second-selection flow. A single lesion crop
    is now sufficient, since segmentation is done properly.
  - Selection state was previously stored as raw canvas coordinates,
    which broke as soon as the image was scrolled/resized. The image is
    now drawn at a fixed, known offset and coordinates are translated
    back into image space explicitly.
  - Results now show all four ABCDE(D) criteria with bands and an
    annotated overlay image, plus buttons to export a JSON/text report.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from PIL import Image, ImageTk

from .analyzer import MoleAnalyzer, AnalysisResult
from .segmentation import SegmentationError
from .report import save_annotated_overlay, save_json_report, save_text_summary

CANVAS_W, CANVAS_H = 800, 600

BAND_COLORS = {
    "Low concern": "#2e7d32",
    "Moderate concern": "#ef6c00",
    "Elevated concern": "#c62828",
}


class MoleAnalyzerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mole Analyzer")
        self.root.geometry("1000x950")
        self.root.configure(bg="#f5f5f5")

        self.analyzer = MoleAnalyzer()
        self.original_image: Image.Image | None = None
        self.display_image_obj: Image.Image | None = None
        self.display_scale = 1.0
        self.photo = None

        self.start_xy: tuple[int, int] | None = None
        self.rect_id: int | None = None
        self.selecting = False
        self.last_result: AnalysisResult | None = None

        self._build_layout()

    # ------------------------------------------------------------------ UI
    def _build_layout(self):
        top_bar = tk.Frame(self.root, bg="#f5f5f5")
        top_bar.pack(pady=10)

        tk.Button(
            top_bar, text="Upload Image", command=self.upload_image,
            font=("Helvetica", 13), bg="#4CAF50", fg="white", padx=10,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            top_bar, text="Analyze Selection", command=self.analyze_selection,
            font=("Helvetica", 13), bg="#1976D2", fg="white", padx=10,
        ).pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.root, width=CANVAS_W, height=CANVAS_H, bg="white", relief="solid", bd=1)
        self.canvas.pack(pady=15)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)

        hint = tk.Label(
            self.root,
            text="Click once to start a selection box around the lesion, click again to finish it.",
            font=("Helvetica", 10), fg="#555555", bg="#f5f5f5",
        )
        hint.pack()

        self.results_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.results_frame.pack(pady=15, fill=tk.X)

        self.disclaimer = tk.Label(
            self.root,
            text=(
                "Educational tool only -- not a medical device and not a diagnosis. "
                "Consult a dermatologist for any lesion of concern."
            ),
            font=("Helvetica", 9, "italic"), fg="#c62828", bg="#f5f5f5", wraplength=900,
        )
        self.disclaimer.pack(pady=(5, 10))

    # ------------------------------------------------------------- upload
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.original_image = Image.open(file_path)
        self._render_to_canvas()
        self._clear_results()

    def _render_to_canvas(self):
        assert self.original_image is not None
        img = self.original_image.copy()
        img.thumbnail((CANVAS_W, CANVAS_H))
        self.display_image_obj = img
        self.display_scale = self.original_image.width / img.width

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.rect_id = None
        self.start_xy = None
        self.selecting = False

    # ----------------------------------------------------------- selection
    def _on_click(self, event):
        if self.display_image_obj is None:
            return
        x = max(0, min(event.x, self.display_image_obj.width))
        y = max(0, min(event.y, self.display_image_obj.height))

        if not self.selecting:
            self.start_xy = (x, y)
            self.selecting = True
            if self.rect_id is not None:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x, y, x, y, outline="red", width=2)
        else:
            self.selecting = False
            self.end_xy = (x, y)
            self.canvas.coords(self.rect_id, *self.start_xy, *self.end_xy)

    def _on_motion(self, event):
        if self.selecting and self.rect_id is not None:
            x = max(0, min(event.x, self.display_image_obj.width))
            y = max(0, min(event.y, self.display_image_obj.height))
            self.canvas.coords(self.rect_id, *self.start_xy, x, y)

    def _get_selection_box_in_original_coords(self) -> tuple[int, int, int, int] | None:
        if self.start_xy is None or not hasattr(self, "end_xy"):
            return None
        (x0, y0), (x1, y1) = self.start_xy, self.end_xy
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return None
        s = self.display_scale
        return (int(x0 * s), int(y0 * s), int(x1 * s), int(y1 * s))

    # ------------------------------------------------------------ analyze
    def analyze_selection(self):
        if self.original_image is None:
            messagebox.showwarning("No image", "Upload an image first.")
            return
        box = self._get_selection_box_in_original_coords()
        if box is None:
            messagebox.showwarning("No selection", "Draw a selection box around the lesion first.")
            return

        crop = self.original_image.crop(box)
        try:
            result = self.analyzer.analyze(crop)
        except SegmentationError as e:
            messagebox.showerror("Segmentation failed", str(e))
            return
        except Exception as e:  # noqa: BLE001 - surface unexpected errors to the user
            messagebox.showerror("Analysis error", f"Unexpected error: {e}")
            return

        self.last_crop = crop
        self.last_result = result
        self._display_results(result)

    def _clear_results(self):
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.last_result = None

    def _display_results(self, result: AnalysisResult):
        self._clear_results()

        header = tk.Label(
            self.results_frame,
            text=f"Overall: {result.overall_band.value} (score {result.overall_score:.2f}/1.00)",
            font=("Helvetica", 16, "bold"),
            fg=BAND_COLORS.get(result.overall_band.value, "#000000"),
            bg="#f5f5f5",
        )
        header.pack(pady=(0, 10))

        for c in result.criteria:
            row = tk.Frame(self.results_frame, bg="#f5f5f5")
            row.pack(fill=tk.X, padx=40, pady=2)
            tk.Label(
                row, text=f"{c.name}:", font=("Helvetica", 12, "bold"), bg="#f5f5f5", width=18, anchor="w"
            ).pack(side=tk.LEFT)
            tk.Label(
                row, text=f"{c.band.value} -- {c.detail}", font=("Helvetica", 12),
                fg=BAND_COLORS.get(c.band.value, "#000000"), bg="#f5f5f5", anchor="w",
            ).pack(side=tk.LEFT)

        btn_row = tk.Frame(self.results_frame, bg="#f5f5f5")
        btn_row.pack(pady=10)
        tk.Button(btn_row, text="Export Report...", command=self._export_report,
                   font=("Helvetica", 11), bg="#455A64", fg="white", padx=8).pack(side=tk.LEFT, padx=5)

    def _export_report(self):
        if self.last_result is None:
            return
        directory = filedialog.askdirectory(title="Choose folder for report output")
        if not directory:
            return
        out_dir = Path(directory)
        save_json_report(self.last_result, out_dir / "mole_report.json")
        save_text_summary(self.last_result, out_dir / "mole_report.txt")
        save_annotated_overlay(self.last_crop, self.last_result, out_dir / "mole_overlay.png")
        messagebox.showinfo("Export complete", f"Report saved to {out_dir}")


def run():
    root = tk.Tk()
    MoleAnalyzerApp(root)
    root.mainloop()
