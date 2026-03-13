import csv
import random
from pathlib import Path

import soundfile as sf

DATA_ROOT = Path(r"/home/caryzxy/datasets/Audio_Speech_Actors_01-24")
OUT_DIR = Path("./ravdess_manifests")
SEED = 42

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def get_duration(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    return float(info.duration)


def parse_actor_id(actor_dir_name: str) -> str:
    # "Actor_01" -> "01"
    return actor_dir_name.split("_")[-1]


def collect_by_speaker(data_root: Path):
    speaker_to_files = {}
    for actor_dir in sorted(data_root.glob("Actor_*")):
        if actor_dir.is_dir():
            spk = parse_actor_id(actor_dir.name)
            wavs = sorted(actor_dir.glob("*.wav"))
            speaker_to_files[spk] = wavs
    return speaker_to_files


def split_list(items, train_ratio, valid_ratio):
    items = items.copy()
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_items = items[:n_train]
    valid_items = items[n_train:n_train + n_valid]
    test_items = items[n_train + n_valid:]
    return train_items, valid_items, test_items


def build_rows(file_list, spk):
    rows = []
    for i, wav_path in enumerate(file_list):
        utt_id = f"{spk}_{i}_{wav_path.stem}"
        duration = get_duration(wav_path)
        rows.append({
            "ID": utt_id,
            "duration": duration,
            "wav": str(wav_path).replace("\\", "/"),
            "spk_id": spk,
        })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "duration", "wav", "spk_id"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    speaker_to_files = collect_by_speaker(DATA_ROOT)

    train_rows, valid_rows, test_rows = [], [], []

    for spk, wavs in speaker_to_files.items():
        train_wavs, valid_wavs, test_wavs = split_list(
            wavs, TRAIN_RATIO, VALID_RATIO
        )
        train_rows.extend(build_rows(train_wavs, spk))
        valid_rows.extend(build_rows(valid_wavs, spk))
        test_rows.extend(build_rows(test_wavs, spk))

    write_csv(OUT_DIR / "train.csv", train_rows)
    write_csv(OUT_DIR / "valid.csv", valid_rows)
    write_csv(OUT_DIR / "test.csv", test_rows)

    print(f"train: {len(train_rows)}")
    print(f"valid: {len(valid_rows)}")
    print(f"test:  {len(test_rows)}")


if __name__ == "__main__":
    main()