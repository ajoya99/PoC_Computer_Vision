from pathlib import Path

from ultralytics import YOLO


data = "data/processed/merged_v1/data.yaml"
small_ckpt = Path("runs/detect/models/yolo26s_high_s_e18_f50/weights/best.pt")
nano_ckpt = Path("runs/detect/models/yolo26n_high_n_e18_f50/weights/best.pt")

if not small_ckpt.exists():
    raise FileNotFoundError(f"Missing {small_ckpt}")
if not nano_ckpt.exists():
    raise FileNotFoundError(f"Missing {nano_ckpt}")


def eval_model(tag: str, ckpt: Path) -> dict:
    model = YOLO(str(ckpt))
    val_res = model.val(
        data=data,
        split="val",
        project="runs/detect/models",
        name=f"{tag}_high_compare_val",
        exist_ok=True,
        verbose=False,
    )
    test_res = model.val(
        data=data,
        split="test",
        project="runs/detect/models",
        name=f"{tag}_high_compare_test",
        exist_ok=True,
        verbose=False,
    )
    names = list(model.names.values())
    val_maps = dict(zip(names, [float(x) for x in val_res.box.maps]))
    test_maps = dict(zip(names, [float(x) for x in test_res.box.maps]))
    return {
        "val": val_res.results_dict,
        "test": test_res.results_dict,
        "val_maps": val_maps,
        "test_maps": test_maps,
    }


def main() -> None:
    small = eval_model("small", small_ckpt)
    nano = eval_model("nano", nano_ckpt)

    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    print("=== HIGH PARAMS COMPARISON (E18 F50 IMG768) ===")
    print("--- OVERALL (TEST) ---")
    for key in metric_keys:
        s = float(small["test"][key])
        n = float(nano["test"][key])
        print("{}: small={:.6f} nano={:.6f} delta_small_minus_nano={:+.6f}".format(key, s, n, s - n))

    print("--- OVERALL (VAL) ---")
    for key in metric_keys:
        s = float(small["val"][key])
        n = float(nano["val"][key])
        print("{}: small={:.6f} nano={:.6f} delta_small_minus_nano={:+.6f}".format(key, s, n, s - n))

    print("--- PER-CLASS mAP50-95 (TEST) ---")
    for cls in small["test_maps"].keys():
        s = small["test_maps"][cls]
        n = nano["test_maps"][cls]
        print("{}: small={:.6f} nano={:.6f} delta={:+.6f}".format(cls, s, n, s - n))

    print("--- PER-CLASS mAP50-95 (VAL) ---")
    for cls in small["val_maps"].keys():
        s = small["val_maps"][cls]
        n = nano["val_maps"][cls]
        print("{}: small={:.6f} nano={:.6f} delta={:+.6f}".format(cls, s, n, s - n))


if __name__ == "__main__":
    main()
