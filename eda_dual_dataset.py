from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)


@dataclass
class DatasetSpec:
    name: str
    root: Path
    color: str
    train_images: Path
    val_images: Path
    test_images: Path


IPHONE = DatasetSpec(
    name="Road\\_poles\\_iPhone",
    root=ROOT / "Poles2025" / "Poles2025" / "Road_poles_iPhone",
    color="#1f4e79",
    train_images=ROOT / "Poles2025" / "Poles2025" / "Road_poles_iPhone" / "data" / "images" / "Train" / "train",
    val_images=ROOT / "Poles2025" / "Poles2025" / "Road_poles_iPhone" / "data" / "images" / "Val" / "val",
    test_images=ROOT / "Poles2025" / "Poles2025" / "Road_poles_iPhone" / "data" / "images" / "Test" / "test",
)

V1 = DatasetSpec(
    name="roadpoles\\_v1",
    root=ROOT / "Poles2025" / "Poles2025" / "roadpoles_v1",
    color="#c47a00",
    train_images=ROOT / "Poles2025" / "Poles2025" / "roadpoles_v1" / "train" / "images",
    val_images=ROOT / "Poles2025" / "Poles2025" / "roadpoles_v1" / "valid" / "images",
    test_images=ROOT / "Poles2025" / "Poles2025" / "roadpoles_v1" / "test" / "images",
)


def image_list(folder: Path) -> list[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file()])


def label_path(image_path: Path) -> Path:
    return Path(image_path.with_suffix(".txt").as_posix().replace("images", "labels"))


def split_stats(folder: Path) -> dict:
    images = image_list(folder)
    areas = []
    aspect_ratios = []
    centers_x = []
    centers_y = []
    boxes_per_image = []
    for img_path in images:
        lbl = label_path(img_path)
        count = 0
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                areas.append(bw * bh * 100.0)
                aspect_ratios.append(bh / max(bw, 1e-9))
                centers_x.append(cx)
                centers_y.append(cy)
                count += 1
        boxes_per_image.append(count)
    return {
        "num_images": len(images),
        "total_boxes": int(sum(boxes_per_image)),
        "boxes_per_image": boxes_per_image,
        "areas_pct": areas,
        "aspect_ratios": aspect_ratios,
        "centers_x": centers_x,
        "centers_y": centers_y,
        "median_area_pct": float(np.median(areas)) if areas else 0.0,
        "mean_boxes_per_image": float(np.mean(boxes_per_image)) if boxes_per_image else 0.0,
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 0.0,
    }


def collect_dataset(spec: DatasetSpec) -> dict:
    return {
        "train": split_stats(spec.train_images),
        "val": split_stats(spec.val_images),
        "test": split_stats(spec.test_images),
    }


def save_split_figure(a: dict, b: dict) -> Path:
    splits = ["train", "val", "test"]
    a_counts = [a[s]["num_images"] for s in splits]
    b_counts = [b[s]["num_images"] for s in splits]
    x = np.arange(len(splits))
    w = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 5.3), dpi=220)
    ax.bar(x - w / 2, a_counts, width=w, color=IPHONE.color, label="Road_poles_iPhone")
    ax.bar(x + w / 2, b_counts, width=w, color=V1.color, label="roadpoles_v1")
    ax.set_xticks(x, ["Train", "Val", "Test"])
    ax.set_ylabel("Images")
    ax.set_title("Dataset sizes by split")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    ymax = max(a_counts + b_counts)
    for xpos, value in zip(x - w / 2, a_counts):
        ax.text(xpos, value + ymax * 0.02, str(value), ha="center", va="bottom")
    for xpos, value in zip(x + w / 2, b_counts):
        ax.text(xpos, value + ymax * 0.02, str(value), ha="center", va="bottom")
    fig.tight_layout()
    out = OUT / "dataset_sizes_by_split.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_geometry_figure(a: dict, b: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5), dpi=220)
    labels = ["Road_poles_iPhone", "roadpoles_v1"]
    colors = [IPHONE.color, V1.color]
    metrics = [
        ("Median box area\n(% of image)", [a["train"]["median_area_pct"], b["train"]["median_area_pct"]]),
        ("Mean height / width", [a["train"]["mean_aspect_ratio"], b["train"]["mean_aspect_ratio"]]),
        ("Mean boxes / image", [a["train"]["mean_boxes_per_image"], b["train"]["mean_boxes_per_image"]]),
    ]
    for ax, (title, values) in zip(axes, metrics):
        ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=10)
        for i, value in enumerate(values):
            fmt = f"{value:.3f}" if value < 1 else f"{value:.2f}"
            ax.text(i, value, fmt, ha="center", va="bottom")
    fig.suptitle("Train-set object geometry summary", fontsize=18, y=1.03)
    fig.tight_layout()
    out = OUT / "object_geometry_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_area_distribution(a: dict, b: dict) -> Path:
    a_vals = np.array(a["train"]["areas_pct"])
    b_vals = np.array(b["train"]["areas_pct"])
    xmax = max(np.percentile(a_vals, 97), np.percentile(b_vals, 97))
    bins = np.linspace(0.0, xmax, 28)
    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=220)
    ax.hist(a_vals, bins=bins, alpha=0.65, color=IPHONE.color, label="Road_poles_iPhone")
    ax.hist(b_vals, bins=bins, alpha=0.60, color=V1.color, label="roadpoles_v1")
    ax.set_xlabel("Bounding box area (% of image)")
    ax.set_ylabel("Count")
    ax.set_title("Train bounding-box area distribution")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT / "bbox_area_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_heatmaps(a: dict, b: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=220)
    datasets = [("Road_poles_iPhone", a["train"]), ("roadpoles_v1", b["train"])]
    image = None

    for ax, (name, stats) in zip(axes, datasets):
        hist, _, _ = np.histogram2d(stats["centers_x"], stats["centers_y"], bins=30, range=[[0, 1], [0, 1]])
        image = ax.imshow(hist.T, extent=[0, 1, 0, 1], origin="lower", cmap="Reds", aspect="equal")
        ax.set_title(name)
        ax.set_xlabel("left -> right")
        ax.set_ylabel("bottom -> top")

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="4.5%", pad=0.18)
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Count per bin")

    fig.suptitle("Pole centre distribution in the train split", fontsize=18, y=1.02)
    fig.subplots_adjust(wspace=0.26, right=0.95)
    out = OUT / "pole_center_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_samples() -> Path:
    def pick_diverse(paths: list[Path]) -> list[Path]:
        if len(paths) <= 2:
            return paths
        idxs = [len(paths) // 6, (5 * len(paths)) // 6]
        return [paths[idxs[0]], paths[idxs[1]]]

    iphone_imgs = pick_diverse(image_list(IPHONE.train_images))
    v1_imgs = pick_diverse(image_list(V1.train_images))
    canvas = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 22), "Representative training images", fill="black")
    draw.text((210, 68), "Road_poles_iPhone", fill=IPHONE.color)
    draw.text((1035, 68), "roadpoles_v1", fill=V1.color)

    def paste_images(paths: list[Path], x0: int, outline: str) -> None:
        x = x0
        for path in paths:
            img = Image.open(path).convert("RGB")
            thumb = ImageOps.fit(img, (330, 720), method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))
            canvas.paste(thumb, (x, 105))
            draw.rectangle((x, 105, x + 330, 825), outline=outline, width=5)
            x += 360

    paste_images(iphone_imgs, 45, IPHONE.color)
    paste_images(v1_imgs, 765, V1.color)
    out = OUT / "representative_training_images.png"
    canvas.save(out)
    return out


def write_summary(a: dict, b: dict) -> Path:
    payload = {
        "Road_poles_iPhone": {
            "train_images": a["train"]["num_images"],
            "val_images": a["val"]["num_images"],
            "median_area_pct": a["train"]["median_area_pct"],
            "mean_boxes_per_image": a["train"]["mean_boxes_per_image"],
            "mean_aspect_ratio": a["train"]["mean_aspect_ratio"],
        },
        "roadpoles_v1": {
            "train_images": b["train"]["num_images"],
            "val_images": b["val"]["num_images"],
            "median_area_pct": b["train"]["median_area_pct"],
            "mean_boxes_per_image": b["train"]["mean_boxes_per_image"],
            "mean_aspect_ratio": b["train"]["mean_aspect_ratio"],
        },
        "takeaways": [
            "Both datasets are sparse, so the problem is dominated by small-object visibility rather than crowding.",
            "roadpoles_v1 is the geometrically harder benchmark because poles are thinner and more elongated.",
            "Road_poles_iPhone is larger, which helps explain why it is easier to train and validate stable models on it.",
            "These differences justify treating the two leaderboards separately when discussing results.",
        ],
    }
    out = OUT / "eda_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = OUT / "eda_summary.md"
    md.write_text(
        "\n".join(
            [
                "# EDA Summary",
                "",
                f"- `Road_poles_iPhone`: {a['train']['num_images']} train / {a['val']['num_images']} val",
                f"- `roadpoles_v1`: {b['train']['num_images']} train / {b['val']['num_images']} val",
                f"- Median box area: {a['train']['median_area_pct']:.3f}% vs {b['train']['median_area_pct']:.3f}%",
                f"- Mean height/width ratio: {a['train']['mean_aspect_ratio']:.2f} vs {b['train']['mean_aspect_ratio']:.2f}",
                f"- Mean boxes/image: {a['train']['mean_boxes_per_image']:.2f} vs {b['train']['mean_boxes_per_image']:.2f}",
                "",
                "## Presentation takeaway",
                "",
                "- The datasets should not be presented as identical tasks.",
                "- `roadpoles_v1` is the harder benchmark because poles are visually thinner and smaller.",
                "- Both datasets support the same modeling intuition: high resolution is more important than handling dense clutter.",
            ]
        ),
        encoding="utf-8",
    )
    return md


def main() -> None:
    iphone = collect_dataset(IPHONE)
    v1 = collect_dataset(V1)
    outputs = [
        save_split_figure(iphone, v1),
        save_geometry_figure(iphone, v1),
        save_area_distribution(iphone, v1),
        save_heatmaps(iphone, v1),
        save_samples(),
        write_summary(iphone, v1),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
