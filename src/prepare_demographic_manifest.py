import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "UTKFace"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "classifiers"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


GENDER_MAP = {
    0: "male",
    1: "female",
}


AGE_GROUP_MAP = {
    0: "child",
    1: "young",
    2: "old",
}


def get_age_group(age):
    if age <= 12:
        return 0

    if age <= 35:
        return 1

    return 2


def parse_utkface_filename(image_path: Path):
    """
    UTKFace filename format:
    age_gender_race_date.jpg

    Example:
    24_0_2_20170116174525125.jpg
    age = 24
    gender = 0
    """

    filename = image_path.name

    # Bazı dosyalar .jpg.chip.jpg olabilir.
    # Biz yine ilk underscore parçalarını okuyacağız.
    parts = filename.split("_")

    if len(parts) < 2:
        return None

    try:
        age = int(parts[0])
        gender = int(parts[1])
    except ValueError:
        return None

    if gender not in GENDER_MAP:
        return None

    if age < 0 or age > 120:
        return None

    age_group = get_age_group(age)

    return {
        "image_path": str(image_path),
        "age": age,
        "gender_label": gender,
        "gender_name": GENDER_MAP[gender],
        "age_group_label": age_group,
        "age_group_name": AGE_GROUP_MAP[age_group],
    }


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"UTKFace folder not found: {RAW_DIR}")

    image_paths = [
        p for p in RAW_DIR.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Found images: {len(image_paths)}")
    print(f"Source folder: {RAW_DIR}")

    rows = []

    for image_path in image_paths:
        parsed = parse_utkface_filename(image_path)

        if parsed is not None:
            rows.append(parsed)

    df = pd.DataFrame(rows)

    print()
    print("Parsed valid images:", len(df))

    print()
    print("Gender distribution:")
    print(df["gender_name"].value_counts())

    print()
    print("Age group distribution:")
    print(df["age_group_name"].value_counts())

    # Stratify için age_group + gender birleşimi kullanıyoruz.
    # Böylece train/val dağılımı daha dengeli olur.
    df["stratify_key"] = df["age_group_name"] + "_" + df["gender_name"]

    train_df, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["stratify_key"]
    )

    # Stratify key eğitimde gerekmiyor.
    train_df = train_df.drop(columns=["stratify_key"])
    val_df = val_df.drop(columns=["stratify_key"])
    df = df.drop(columns=["stratify_key"])

    all_csv = OUTPUT_DIR / "demographic_all.csv"
    train_csv = OUTPUT_DIR / "demographic_train.csv"
    val_csv = OUTPUT_DIR / "demographic_val.csv"

    df.to_csv(all_csv, index=False)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    print()
    print("==============================")
    print("Manifest files created.")
    print("==============================")
    print(f"All:   {all_csv}")
    print(f"Train: {train_csv}")
    print(f"Val:   {val_csv}")

    print()
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))

    print()
    print("Train gender distribution:")
    print(train_df["gender_name"].value_counts())

    print()
    print("Val gender distribution:")
    print(val_df["gender_name"].value_counts())

    print()
    print("Train age group distribution:")
    print(train_df["age_group_name"].value_counts())

    print()
    print("Val age group distribution:")
    print(val_df["age_group_name"].value_counts())


if __name__ == "__main__":
    main()
