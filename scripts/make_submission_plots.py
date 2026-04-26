from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "training_curves"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_loss() -> None:
    rows = read_csv(OUT / "run4b_training_loss.csv")
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]

    width, height = 1100, 660
    margin_left, margin_right, margin_top, margin_bottom = 110, 45, 70, 105
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x_min, x_max = min(steps), max(steps)
    y_values = [max(loss, 1e-5) for loss in losses]
    y_min, y_max = -5.0, 0.1

    def x_pos(step: int) -> float:
        plot_w = width - margin_left - margin_right
        return margin_left + (step - x_min) / (x_max - x_min) * plot_w

    def y_pos(loss: float) -> float:
        import math

        plot_h = height - margin_top - margin_bottom
        log_loss = math.log10(max(loss, 1e-5))
        return margin_top + (y_max - log_loss) / (y_max - y_min) * plot_h

    _draw_axes(draw, width, height, margin_left, margin_right, margin_top, margin_bottom)
    draw.text((margin_left, 26), "Run4b Qwen3-8B QLoRA SFT Training Loss", fill="#1a202c", font=font)
    draw.text((width // 2 - 45, height - 38), "training step", fill="#1a202c", font=font)
    draw.text((18, height // 2), "SFT loss (log10 scale)", fill="#1a202c", font=font)

    for tick in [20, 60, 100, 140, 180, 220]:
        x = x_pos(tick)
        draw.line([(x, margin_top), (x, height - margin_bottom)], fill="#edf2f7")
        draw.text((x - 10, height - margin_bottom + 12), str(tick), fill="#4a5568", font=font)
    for label, log_value in [("1", 0), ("0.1", -1), ("0.01", -2), ("0.001", -3), ("1e-4", -4), ("1e-5", -5)]:
        y = margin_top + (y_max - log_value) / (y_max - y_min) * (height - margin_top - margin_bottom)
        draw.line([(margin_left, y), (width - margin_right, y)], fill="#edf2f7")
        draw.text((margin_left - 58, y - 6), label, fill="#4a5568", font=font)

    points = [(x_pos(step), y_pos(loss)) for step, loss in zip(steps, losses)]
    draw.line(points, fill="#2b6cb0", width=4)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#2b6cb0")
    draw.text((margin_left, height - 70), "Source: HF job 69edb014d2c8bd8662bcf5ba logs; final train_loss=0.0565.", fill="#4a5568", font=font)
    image.save(OUT / "run4b_training_loss.png")


def plot_rewards() -> None:
    rows = read_csv(OUT / "run4b_eval_rewards.csv")
    labels = [row["agent"] for row in rows]
    rewards = [float(row["avg_reward"]) for row in rows]
    primary = [float(row["primary_reward"]) for row in rows]

    width, height = 1200, 680
    margin_left, margin_right, margin_top, margin_bottom = 90, 35, 72, 150
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    _draw_axes(draw, width, height, margin_left, margin_right, margin_top, margin_bottom)
    draw.text((margin_left, 26), "Run4b Held-Out Reward vs Baselines", fill="#1a202c", font=font)
    draw.text((width // 2 - 70, height - 34), "agent / checkpoint", fill="#1a202c", font=font)
    draw.text((18, height // 2), "held-out score", fill="#1a202c", font=font)

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    group_w = plot_w / len(rows)
    bar_w = min(46, group_w * 0.32)

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + (1.0 - tick) * plot_h
        draw.line([(margin_left, y), (width - margin_right, y)], fill="#edf2f7")
        draw.text((margin_left - 42, y - 6), f"{tick:.2f}", fill="#4a5568", font=font)

    for idx, (label, reward, primary_score) in enumerate(zip(labels, rewards, primary)):
        center = margin_left + group_w * idx + group_w / 2
        for offset, value, color in [(-bar_w / 1.7, reward, "#2f855a"), (bar_w / 1.7, primary_score, "#805ad5")]:
            x0 = center + offset - bar_w / 2
            x1 = center + offset + bar_w / 2
            y0 = margin_top + (1.0 - value) * plot_h
            y1 = height - margin_bottom
            draw.rectangle((x0, y0, x1, y1), fill=color)
            draw.text((x0, y0 - 15), f"{value:.2f}", fill="#1a202c", font=font)
        draw.text((center - 48, height - margin_bottom + 15), label.replace("_", "\n"), fill="#4a5568", font=font)

    draw.rectangle((margin_left, height - 86, margin_left + 14, height - 72), fill="#2f855a")
    draw.text((margin_left + 20, height - 88), "avg reward", fill="#1a202c", font=font)
    draw.rectangle((margin_left + 115, height - 86, margin_left + 129, height - 72), fill="#805ad5")
    draw.text((margin_left + 135, height - 88), "primary reward", fill="#1a202c", font=font)
    draw.text((margin_left, height - 58), "Source: 150-seed eval_150 for run4b plus previous run3 eval summary.", fill="#4a5568", font=font)
    image.save(OUT / "run4b_eval_rewards.png")


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    margin_left: int,
    margin_right: int,
    margin_top: int,
    margin_bottom: int,
) -> None:
    draw.line([(margin_left, margin_top), (margin_left, height - margin_bottom)], fill="#2d3748", width=2)
    draw.line([(margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom)], fill="#2d3748", width=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_loss()
    plot_rewards()


if __name__ == "__main__":
    main()
