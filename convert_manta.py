import os
from pathlib import Path
import json
import cv2
import numpy as np


""" 
    This script is used to convert the vanilla MANTA(-Tiny) data set from its regular format (5 views in one image) to the 
    Real-IAD compatible format (all views in separate image files)
"""


# TODO: change this to the path of your MANTA(-Tiny) data set
base_directory = Path("MANTA-Tiny/")

# TODO: change this to the path you want your Real-IAD-style data set to be saved
save_to_base = Path("MANTA-Tiny_realiad/")

for meta_category in ["agriculture", "electronics", "groceries", "mechanics", "medicine"]:

    cat_path = base_directory / meta_category
    categories = os.listdir(cat_path)
    for category in categories:
        print(f"Converting '{category}'")

        json_data = {
            "meta": {
                "prefix": f"{category}",
                "normal_class": "OK",
                "pre_transform": False
            },
            "train": [],
            "test": []
        }

        # iterate over train images
        print("copying train set")
        train_base_path = Path(cat_path / category / "train" / "good")
        save_to_good = Path(save_to_base, "classes", category, "OK")
        for img_idx, img_path in enumerate(train_base_path.rglob("*")):
            img = cv2.imread(img_path)
            stem = img_path.stem
            H, W, _ = img.shape
            split_width = W // 5
            images = [img[:, i*split_width:(i+1)*split_width]
                      for i in range(5)]

            save_to_instance = Path(save_to_good, f"S{img_idx + 1:04d}")
            os.makedirs(save_to_instance, exist_ok=True)

            for cam_idx in range(5):
                final_img_path = Path(
                    save_to_instance, f"{stem}_C{cam_idx + 1}.png")
                cv2.imwrite(str(final_img_path), images[cam_idx])
                json_data["train"].append({
                    "category": category,
                    "anomaly_class": "OK",
                    "image_path": str(Path(*final_img_path.parts[-3:])),
                    "mask_path": None,
                })

        print("copying test set")
        # iterate over test categories
        test_base_path = Path(cat_path / category / "test")
        for test_class in test_base_path.glob("*"):
            print(f"executing {category} with class {test_class}")

            class_name = test_class.stem
            is_ano = class_name != "good"

            save_to_test = Path(save_to_base, "classes",
                                category, "NG", class_name)

            for img_idx, img_path in enumerate(test_class.rglob("*")):

                img = cv2.imread(img_path)
                stem = img_path.stem
                H, W, _ = img.shape
                split_width = W // 5
                images = [img[:, i*split_width:(i+1)*split_width]
                          for i in range(5)]

                if is_ano:
                    masks_base_path = cat_path / category / \
                        "ground_truth" / class_name / f"{stem}.png"
                    mask = cv2.imread(masks_base_path)
                    H_mask, W_mask, _ = img.shape
                    assert H_mask == H and W == W_mask
                    all_masks = [
                        mask[:, i*split_width:(i+1)*split_width] for i in range(5)]

                save_to_instance = Path(save_to_test, f"S{img_idx + 1:04d}")
                os.makedirs(save_to_instance, exist_ok=True)

                for cam_idx in range(5):

                    # save single-view image
                    final_img_path = Path(
                        save_to_instance, f"{stem}_C{cam_idx + 1}.png")
                    cv2.imwrite(str(final_img_path), images[cam_idx])

                    if is_ano:
                        final_mask_path = Path(
                            save_to_instance, f"{stem}_C{cam_idx + 1}_mask.png")
                        cv2.imwrite(str(final_mask_path), all_masks[cam_idx])
                        is_current_image_ano = np.sum(all_masks[cam_idx]) > 0
                    else:
                        is_current_image_ano = False
                    # append to json
                    json_data["test"].append({
                        "category": category,
                        "anomaly_class": class_name if is_current_image_ano else "OK",  # DETERMINE FROM MASK
                        "image_path": str(Path(*final_img_path.parts[-4:])),
                        "mask_path": str(Path(*final_mask_path.parts[-4:])),
                    })

        print("saving json")
        json_dir = save_to_base / "jsons" / "realiad_jsons"
        os.makedirs(json_dir, exist_ok=True)
        with open(json_dir / f"{category}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)
