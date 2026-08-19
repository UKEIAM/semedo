import json
import ast
import os
from typing import Any, Iterable
from pathlib import Path
from PIL import Image, ImageOps
import pandas as pd
import numpy as np


fig: Any = None
ax: Any = None
df: pd.DataFrame = pd.DataFrame()
image_files = []
current_index = 0
min_points = 4
max_points = 50
output_csv: Any = None
points = []
scatter_points = []
prev_button: Any = None
next_button: Any = None
skip_button: Any = None
undo_button: Any = None


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def init_context(**kwargs):
    """Injects runtime context values into module-level globals."""
    for name, value in kwargs.items():
        globals()[name] = value


def _resolve_path_value(path_value, base_dir: Path) -> Path:
    """Resolve a possibly relative path against a base directory."""
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def find_annotation_config_path(config_path=None) -> Path:
    """Find the annotation config file, preferring explicit and local paths."""
    if config_path is not None:
        candidate = Path(config_path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Annotation config not found: {candidate}")

    candidates = [
        Path.cwd() / "annotation_config.json",
        Path.cwd() / "src" / "annotate" / "annotation_config.json",
        Path(__file__).resolve().parent / "annotation_config.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No annotation_config.json found. Please create one in src/annotate/ or pass an explicit path."
    )


def load_annotation_config(config_path=None) -> dict:
    """Load and normalize the annotation JSON config."""
    config_file = find_annotation_config_path(config_path)
    with open(config_file, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    base_dir = config_file.parent
    image_folders = raw_config.get("image_folders")
    if not image_folders and raw_config.get("data_dir"):
        image_folders = [raw_config["data_dir"]]
    if not image_folders:
        raise ValueError(
            "The annotation config needs either 'image_folders' or 'data_dir'."
        )

    image_extensions = raw_config.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)
    image_extensions = tuple(
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in image_extensions
    )

    output_csv_value = raw_config.get("output_csv") or raw_config.get("gt_output_csv")
    if not output_csv_value:
        raise ValueError(
            "The annotation config needs an 'output_csv' or 'gt_output_csv' path."
        )

    resolved_image_folders = [
        _resolve_path_value(folder, base_dir) for folder in image_folders
    ]
    output_csv = _resolve_path_value(output_csv_value, base_dir)

    return {
        "config_path": config_file,
        "base_dir": base_dir,
        "raw_config": raw_config,
        "image_folders": resolved_image_folders,
        "output_csv": output_csv,
        "min_points": int(raw_config.get("min_points", 4)),
        "max_points": int(raw_config.get("max_points", 50)),
        "image_extensions": image_extensions,
    }


def collect_image_files(
    image_folders: Iterable[Path], image_extensions=DEFAULT_IMAGE_EXTENSIONS
):
    """Collect and sort supported image files from one or more folders."""
    normalized_extensions = {
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in image_extensions
    }
    files = []
    for folder in image_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Image folder not found: {folder_path}")
        files.extend(
            [
                file_path
                for file_path in folder_path.iterdir()
                if file_path.is_file()
                and file_path.suffix.lower() in normalized_extensions
            ]
        )
    return sorted(files)


def load_annotation_dataframe(output_csv: Path, base_columns):
    """Load the annotation CSV or create an empty dataframe with the expected columns."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        try:
            if output_csv.stat().st_size == 0:
                raise pd.errors.EmptyDataError("Empty CSV file")
            df_local = pd.read_csv(output_csv)
        except pd.errors.EmptyDataError:
            print("Note: CSV is empty. Starting with a new table.")
            df_local = pd.DataFrame(columns=base_columns)
        except Exception as exc:
            print(f"Warning: CSV could not be read ({exc}). Starting with a new table.")
            df_local = pd.DataFrame(columns=base_columns)

        for col in base_columns:
            if col not in df_local.columns:
                df_local[col] = None
        return df_local[base_columns]

    return pd.DataFrame(columns=base_columns)


def clear_points_on_plot():
    """Clears in-memory point state and plotted point handles."""
    global points, scatter_points
    points = []
    scatter_points = []


def draw_existing_points(existing_points):
    """Draws previously saved points and restores them into current state."""
    global points, scatter_points
    points = existing_points.copy()
    scatter_points = []
    for x, y in points:
        scatter = ax.plot(x, y, "ro")
        scatter_points.append(scatter)


def corners_from_row(row):
    """Extract polygon points from a CSV row using polygon_points or legacy corner columns."""
    if "polygon_points" in row.index and pd.notna(row["polygon_points"]):
        try:
            loaded = json.loads(str(row["polygon_points"]))
        except Exception:
            try:
                loaded = ast.literal_eval(str(row["polygon_points"]))
            except Exception:
                loaded = None
        if loaded is not None:
            points = np.array(loaded, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
                return points

    corner_cols = ["tl_x", "tl_y", "bl_x", "bl_y", "br_x", "br_y", "tr_x", "tr_y"]
    if all(col in row.index for col in corner_cols):
        vals = [row[col] for col in corner_cols]
        if all(pd.notna(v) for v in vals):
            return np.array(
                [
                    [float(row["tl_x"]), float(row["tl_y"])],
                    [float(row["bl_x"]), float(row["bl_y"])],
                    [float(row["br_x"]), float(row["br_y"])],
                    [float(row["tr_x"]), float(row["tr_y"])],
                ],
                dtype=np.float32,
            )

    raise ValueError("No GT polygon information found in row.")


def polygon_to_mask(polygon, h, w):
    """Convert a polygon to a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.round(np.array(polygon, dtype=np.float32)).astype(np.int32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return mask
    cv2 = None
    try:
        import cv2 as _cv2

        cv2 = _cv2
    except Exception:
        return mask
    cv2.fillPoly(mask, [poly], 1)
    return mask


def resolve_image_path(raw_path, image_root=None):
    """Resolve a raw/relative image path to an absolute path, tolerating .jpg/.jpeg case variants."""
    if raw_path is None:
        return None
    p = str(raw_path).strip()
    if len(p) == 0:
        return None

    def _with_alt_ext(path_str):
        cands = [path_str]
        lower = path_str.lower()
        if lower.endswith(".jpg"):
            cands.append(path_str[:-4] + ".jpeg")
            cands.append(path_str[:-4] + ".JPG")
            cands.append(path_str[:-4] + ".JPEG")
        elif lower.endswith(".jpeg"):
            cands.append(path_str[:-5] + ".jpg")
            cands.append(path_str[:-5] + ".JPG")
            cands.append(path_str[:-5] + ".JPEG")
        return cands

    # absolute path
    if os.path.isabs(p):
        for cand in _with_alt_ext(p):
            if os.path.isfile(cand):
                return os.path.abspath(cand)

    # relative to provided image_root
    if image_root is not None:
        base_candidate = os.path.abspath(os.path.join(str(image_root), p))
        for cand in _with_alt_ext(base_candidate):
            if os.path.isfile(cand):
                return os.path.abspath(cand)

    # last chance: raw path from cwd
    for cand in _with_alt_ext(p):
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    return None


def get_points_from_df(img_path):
    """Returns saved points for an image from CSV data, with legacy fallback."""
    resolved_current = resolve_image_path(img_path)
    if resolved_current is None:
        # fallback to string comparison
        resolved_current = str(img_path)

    def _match_row(p):
        try:
            rp = resolve_image_path(p)
            if rp is None:
                return str(p) == resolved_current
            return rp == resolved_current
        except Exception:
            return False

    matches = df[df["image_path"].apply(_match_row)]
    if matches.empty:
        return []

    row = matches.iloc[-1]
    try:
        loaded = corners_from_row(row)
        return [(float(x), float(y)) for x, y in loaded][:max_points]
    except Exception:
        return []


def display_image_name(img_path) -> str:
    """Shortens dummy-UKE filenames (pat_<n>__<tag>__<stem>) to just the trailing stem for display."""
    name = Path(img_path).name
    return name.rsplit("__", 1)[-1]


def show_image(index):
    """Displays the current image and loads existing points from the CSV."""
    global points, scatter_points
    if "ax" not in globals() or "fig" not in globals():
        raise RuntimeError(
            "Plot context not initialized. Call init_context(fig=..., ax=..., ...) first."
        )
    ax.clear()
    clear_points_on_plot()
    img_path = image_files[index]
    img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    model_width, model_height = img_pil.size
    img = np.array(img_pil)
    ax.imshow(img)
    ax.set_title(f"{display_image_name(img_path)} - min {min_points}, max {max_points} points")

    existing_points = get_points_from_df(img_path)
    if existing_points:
        mask = polygon_to_mask(existing_points, model_height, model_width)
        ax.imshow(
            mask, cmap="Reds", alpha=0.35, interpolation="nearest", vmin=0, vmax=1
        )
        ax.contour(mask, levels=[0.5], colors="yellow", linewidths=2)
        draw_existing_points(existing_points)

    fig.canvas.draw_idle()
    return model_width, model_height


def redraw_current_image(with_mask=True):
    """Redraws the current image using the in-memory points list."""
    global scatter_points
    if "ax" not in globals() or "fig" not in globals():
        raise RuntimeError(
            "Plot context not initialized. Call init_context(fig=..., ax=..., ...) first."
        )
    img_path = image_files[current_index]
    img_pil = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    model_width, model_height = img_pil.size
    img = np.array(img_pil)

    ax.clear()
    ax.imshow(img)
    ax.set_title(f"{display_image_name(img_path)} - min {min_points}, max {max_points} points")

    if with_mask and len(points) >= 3:
        mask = polygon_to_mask(points, model_height, model_width)
        ax.imshow(
            mask, cmap="Reds", alpha=0.35, interpolation="nearest", vmin=0, vmax=1
        )
        ax.contour(mask, levels=[0.5], colors="yellow", linewidths=2)

    scatter_points = []
    for x, y in points:
        scatter = ax.plot(x, y, "ro")
        scatter_points.append(scatter)

    fig.canvas.draw_idle()
    return model_width, model_height


def save_current_image_points(silent=False):
    """Saves or updates the current image points in the CSV file."""
    global df
    img_path = image_files[current_index]
    model_width, model_height = Image.open(img_path).size

    # Compatibility: keep writing the first 4 points into tl/bl/br/tr too
    four_points = points[:4] + [(None, None)] * (4 - len(points[:4]))

    new_row = {
        "image_path": str(img_path),
        "model_width": model_width,
        "model_height": model_height,
        "polygon_points": json.dumps(points),
        "tl_x": four_points[0][0],
        "tl_y": four_points[0][1],
        "bl_x": four_points[1][0],
        "bl_y": four_points[1][1],
        "br_x": four_points[2][0],
        "br_y": four_points[2][1],
        "tr_x": four_points[3][0],
        "tr_y": four_points[3][1],
    }

    df = df[df["image_path"] != str(img_path)]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(output_csv, index=False)
    if not silent:
        print(
            f"Coordinates for {img_path.name} saved ({len(points)} points) -> {output_csv}"
        )


def onclick(event):
    """Adds a point on mouse click and persists the current annotation."""
    global points, scatter_points
    if event.xdata is None or event.ydata is None:
        return
    if len(points) < max_points:
        x, y = event.xdata, event.ydata
        points.append((x, y))
        save_current_image_points(silent=True)
        redraw_current_image(with_mask=True)
        print(f"Point {len(points)}: ({x:.2f}, {y:.2f}) — automatically saved")
    else:
        print(f"Maximum {max_points} points allowed.")


def on_undo(b):
    """Removes the most recently added point and saves immediately."""
    global points, scatter_points
    if not points:
        print("No point to delete.")
        return
    points.pop()
    save_current_image_points(silent=True)
    redraw_current_image(with_mask=True)
    print(f"Last point deleted. Still {len(points)} point(s). Automatically saved.")


def on_prev(b):
    """Moves to the previous image without forcing a save."""
    global current_index
    if current_index == 0:
        print("You are already at the first image.")
        return
    current_index -= 1
    show_image(current_index)


def on_next(b):
    """Validates minimum points, saves, and advances to the next image."""
    global current_index
    if len(points) < min_points:
        print(f"Please set at least {min_points} points or use 'Skip' to proceed.")
        return

    save_current_image_points()

    current_index += 1
    if current_index >= len(image_files):
        print("All images annotated!")
        next_button.disabled = True
        prev_button.disabled = True
        undo_button.disabled = True
        skip_button.disabled = True
        return

    show_image(current_index)


def on_skip(b):
    """Skips the current image without saving and advances."""
    global current_index
    current_index += 1
    if current_index >= len(image_files):
        print("No more images to skip.")
        next_button.disabled = True
        prev_button.disabled = True
        undo_button.disabled = True
        skip_button.disabled = True
        return
    print("Image skipped.")
    show_image(current_index)
