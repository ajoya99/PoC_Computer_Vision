from __future__ import annotations

import argparse
import base64
import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOMAINS = ("warehouse", "logistic", "mobility", "wheelchair")


def infer_domain_from_name(name: str) -> str | None:
    n = name.lower()
    if n.startswith("warehouse"):
        return "warehouse"
    if n.startswith("logistic"):
        return "logistic"
    if n.startswith("mobility_aids") or n.startswith("mobility"):
        return "mobility"
    if n.startswith("wheelchair"):
        return "wheelchair"
    return None


def gather_dataset_by_domain(dataset_root: Path) -> dict[str, list[dict]]:
    by_domain: dict[str, list[dict]] = {k: [] for k in DOMAINS}
    for split in ("train", "val", "test"):
        img_dir = dataset_root / split / "images"
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            domain = infer_domain_from_name(img_path.name)
            if domain is None:
                continue
            by_domain[domain].append({"image": img_path, "split": split, "domain": domain})
    for domain in DOMAINS:
        if not by_domain[domain]:
            raise RuntimeError(f"No dataset images found for domain: {domain}")
    return by_domain


def pick_dataset_pool(by_domain: dict[str, list[dict]], per_domain: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    picked: list[dict] = []
    for domain in DOMAINS:
        rows = by_domain[domain]
        k = min(per_domain, len(rows))
        picked.extend(rng.sample(rows, k))
    rng.shuffle(picked)
    return picked


def pick_unlabeled_pool(raw_dir: Path, count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    imgs = [p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(imgs) < count:
        raise RuntimeError(f"Need at least {count} images in {raw_dir}, found {len(imgs)}")
    return rng.sample(imgs, count)


def copy_inputs(items: list[Path], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for idx, src in enumerate(items, start=1):
        dst = target_dir / f"{idx:03d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def run_predict(model: YOLO, source_dir: Path, out_root: Path, run_name: str, conf: float, imgsz: int, device: str) -> Path:
    model.predict(
        source=str(source_dir.resolve()),
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(out_root.resolve()),
        name=run_name,
        exist_ok=True,
    )
    return out_root / run_name


def path_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".png":
        mime = "image/png"
    elif suffix == ".bmp":
        mime = "image/bmp"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_html(output_html: Path, tab1_pool: list[dict], tab2_pool: list[dict], slots: int) -> None:
    payload = {
        "tab1": tab1_pool,
        "tab2": tab2_pool,
    }
    payload_js = json.dumps(payload, ensure_ascii=True)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Detection Review UI</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf2;
      --ink: #1f2933;
      --muted: #6b7280;
      --accent: #0f766e;
      --border: #eadfce;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: \"Segoe UI\", Tahoma, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 85% 10%, #f5cf8f 0%, transparent 25%),
        radial-gradient(circle at 10% 90%, #b4e0da 0%, transparent 25%),
        var(--bg);
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 16px; }}
    .tabs {{ display: flex; gap: 10px; margin-bottom: 14px; }}
    .tab-btn {{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--ink);
      padding: 10px 14px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
    }}
    .tab-btn.active {{ background: var(--accent); color: white; border-color: var(--accent); }}
    .panel {{
      display: none;
      background: rgba(255, 250, 242, 0.88);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
    }}
    .panel.active {{ display: block; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      grid-template-rows: repeat(2, auto);
      gap: 12px;
      margin-top: 10px;
    }}
    .card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      background: white;
    }}
    .card img {{ width: 100%; height: 300px; object-fit: contain; display: block; background: #111; }}
    .meta {{ padding: 10px; color: var(--muted); font-size: 0.9rem; }}
    .actions {{ display: flex; justify-content: flex-end; padding: 0 10px 10px; }}
    .next-btn {{
      border: 1px solid var(--accent);
      color: var(--accent);
      background: #ecfdf5;
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Latest Model Visual Review</h1>
    <div class=\"subtitle\">Click Next in any card to refresh only that picture with another predicted sample.</div>

    <div class=\"tabs\">
      <button class=\"tab-btn active\" data-tab=\"tab1\">Dataset Samples</button>
      <button class=\"tab-btn\" data-tab=\"tab2\">Unlabeled AMS/ZWI</button>
    </div>

    <section id=\"tab1\" class=\"panel active\">
      <div class=\"grid\" id=\"grid-tab1\"></div>
    </section>

    <section id=\"tab2\" class=\"panel\">
      <div class=\"grid\" id=\"grid-tab2\"></div>
    </section>
  </div>

  <script>
    const pools = {payload_js};
    const slots = {slots};
    const state = {{
      tab1: Array.from({{ length: slots }}, (_, i) => i % pools.tab1.length),
      tab2: Array.from({{ length: slots }}, (_, i) => i % pools.tab2.length),
    }};

    function cardHtml(tab, slot) {{
      const idx = state[tab][slot];
      const item = pools[tab][idx];
      return `
        <div class=\"card\">
          <img src=\"${{item.src}}\" alt=\"sample\" />
          <div class=\"meta\">${{item.caption}}</div>
          <div class=\"actions\">
            <button class=\"next-btn\" onclick=\"nextImage('${{tab}}', ${{slot}})\">Next</button>
          </div>
        </div>
      `;
    }}

    function renderTab(tab) {{
      const grid = document.getElementById(`grid-${{tab}}`);
      grid.innerHTML = \"\";
      for (let i = 0; i < slots; i++) {{
        grid.innerHTML += cardHtml(tab, i);
      }}
    }}

    function nextImage(tab, slot) {{
      const len = pools[tab].length;
      state[tab][slot] = (state[tab][slot] + 1) % len;
      renderTab(tab);
    }}

    document.querySelectorAll('.tab-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
      }});
    }});

    renderTab('tab1');
    renderTab('tab2');
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def make_tab1_pool(pred_images: list[Path], picked_meta: list[dict]) -> list[dict]:
    if len(pred_images) != len(picked_meta):
        raise RuntimeError("Tab1 pool size mismatch between predictions and metadata")
    out: list[dict] = []
    for img, meta in zip(pred_images, picked_meta):
        caption = f"domain={meta['domain']} | split={meta['split']} | file={meta['image'].name}"
        out.append({"src": path_to_data_uri(img), "caption": caption})
    return out


def make_tab2_pool(pred_images: list[Path], picked_raw: list[Path]) -> list[dict]:
    if len(pred_images) != len(picked_raw):
        raise RuntimeError("Tab2 pool size mismatch between predictions and metadata")
    out: list[dict] = []
    for img, raw in zip(pred_images, picked_raw):
        caption = f"raw={raw.name}"
        out.append({"src": path_to_data_uri(img), "caption": caption})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2-tab visual review UI with per-card Next refresh.")
    parser.add_argument("--weights", default="runs/detect/models/yolo26s_high_s_e18_f50/weights/best.pt")
    parser.add_argument("--dataset-root", default="data/processed/merged_v1")
    parser.add_argument("--raw-dir", default="data/external_scenarios/raw")
    parser.add_argument("--output", default="data/external_scenarios/ui_review")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--dataset-per-domain", type=int, default=8)
    parser.add_argument("--unlabeled-pool", type=int, default=24)
    args = parser.parse_args()

    weights = Path(args.weights)
    dataset_root = Path(args.dataset_root)
    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw folder not found: {raw_dir}")

    by_domain = gather_dataset_by_domain(dataset_root)
    dataset_pool = pick_dataset_pool(by_domain, per_domain=args.dataset_per_domain, seed=args.seed)
    unlabeled_pool = pick_unlabeled_pool(raw_dir, count=args.unlabeled_pool, seed=args.seed + 1)

    dataset_input = output_root / "inputs" / "dataset"
    unlabeled_input = output_root / "inputs" / "unlabeled"
    preds_root = output_root / "predictions"

    if output_root.exists():
        shutil.rmtree(output_root)
    dataset_input.mkdir(parents=True, exist_ok=True)
    unlabeled_input.mkdir(parents=True, exist_ok=True)

    copied_dataset = copy_inputs([r["image"] for r in dataset_pool], dataset_input)
    copied_unlabeled = copy_inputs(unlabeled_pool, unlabeled_input)

    model = YOLO(str(weights))
    tab1_pred_dir = run_predict(
        model=model,
        source_dir=dataset_input,
        out_root=preds_root,
        run_name="tab1_dataset_pool",
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    tab2_pred_dir = run_predict(
        model=model,
        source_dir=unlabeled_input,
        out_root=preds_root,
        run_name="tab2_unlabeled_pool",
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    tab1_pred_images = sorted([p for p in tab1_pred_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    tab2_pred_images = sorted([p for p in tab2_pred_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    if len(tab1_pred_images) < args.slots:
        raise RuntimeError(f"Need at least {args.slots} tab1 predictions, found {len(tab1_pred_images)}")
    if len(tab2_pred_images) < args.slots:
        raise RuntimeError(f"Need at least {args.slots} tab2 predictions, found {len(tab2_pred_images)}")

    tab1_pool = make_tab1_pool(tab1_pred_images, dataset_pool)
    tab2_pool = make_tab2_pool(tab2_pred_images, unlabeled_pool)

    html_path = output_root / "index.html"
    build_html(output_html=html_path, tab1_pool=tab1_pool, tab2_pool=tab2_pool, slots=args.slots)

    print("UI generated:", html_path.resolve())
    print("tab1 pool size:", len(tab1_pool))
    print("tab2 pool size:", len(tab2_pool))
    print("dataset copied:", len(copied_dataset))
    print("unlabeled copied:", len(copied_unlabeled))


if __name__ == "__main__":
    main()
