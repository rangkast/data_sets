import os
import cv2
import json
import glob
import random
import numpy as np
import albumentations as A

GEN_CNT = 2000
DO_GENERATES = True

def create_target_images_with_border(target_dir, generate_dir, size=(50, 50), black_border_thickness=2, background_border_thickness=10):
    os.makedirs(generate_dir, exist_ok=True)
    target_files = sorted(glob.glob("{}/*.jpg".format(target_dir)))
    
    for target_file in target_files:
        img = cv2.imread(target_file)
        resized_img = cv2.resize(img, size)
        
        # Add black border
        bordered_img = cv2.copyMakeBorder(resized_img, black_border_thickness, black_border_thickness, black_border_thickness, black_border_thickness, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        target_name = os.path.basename(target_file)
        cv2.imwrite(os.path.join(generate_dir, target_name), bordered_img)

def apply_augmentation(image):
    transform = A.Compose([
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.Blur(blur_limit=7, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast(p=0.3),
            A.RandomGamma(p=0.3),
        ], p=0.3),
        A.HueSaturationValue(p=0.35),
        A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(192, 197, 191)),  # Rotation limit reduced to 15 degrees
        A.OneOf([
            A.Perspective(scale=(0.02, 0.045), p=0.3),  # Reduced perspective scale
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-7, 7), shear=(-5, 5), p=0.5),  # Reduced affine transformations
        ], p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),  # Adjust hue, saturation, and brightness
    ])

    transformed = transform(image=image)

    return transformed['image']


def is_too_close(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2
    cx2, cy2 = (bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2
    
    distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    
    width1, height1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    width2, height2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    
    min_distance = (width1 + width2) / 2 + (height1 + height2) / 2
    
    return distance < min_distance

def generate_fhd_image_with_targets(image_index, target_generate_dir, output_dir, backgrounds_dir, num_targets=10, image_size=(1920, 1080), target_size=(50, 50), border_thickness=2, padding=20):
    target_files = sorted(glob.glob("{}/*.jpg".format(target_generate_dir)))
    background_files = sorted(glob.glob("{}/*.jpg".format(backgrounds_dir)))
    
    if len(target_files) < num_targets:
        raise ValueError("Not enough target images to generate {} targets.".format(num_targets))
    
    if len(background_files) == 0:
        raise ValueError("No background images found in the backgrounds directory.")
    
    positions = []
    annotations = []

    # 무작위 배경 이미지 선택
    background_image_path = random.choice(background_files)
    background_image = cv2.imread(background_image_path)
    if background_image.shape[:2] != image_size:
        background_image = cv2.resize(background_image, image_size)
    
    # Create a background image with the selected background
    image = background_image.copy()
    
    for i in range(num_targets):
        attempts = 0
        while attempts < 100:  # Limit the number of attempts to find a non-overlapping position
            max_x = image_size[0] - target_size[0] - 2 * border_thickness - padding
            max_y = image_size[1] - target_size[1] - 2 * border_thickness - padding
            x = random.randint(padding, max_x)
            y = random.randint(padding, max_y)
            bbox = (x, y, x + target_size[0] + 4 * border_thickness, y + target_size[1] + 4 * border_thickness)
            
            # Check for overlap using center point distance
            overlap = any(is_too_close(bbox, pos) for pos in positions)
            if not overlap:
                positions.append(bbox)
                target_img = cv2.imread(target_files[i])
                
                # Apply augmentation
                augmented_img = apply_augmentation(target_img)
                
                # Randomly scale the image between -25% and +150%
                scale_factor = random.uniform(0.75, 2.5)
                new_size = (int((target_size[0] + 4 * border_thickness) * scale_factor), int((target_size[1] + 4 * border_thickness) * scale_factor))
                resized_augmented_img = cv2.resize(augmented_img, new_size)
                
                # Ensure augmented image fits in the original bounding box
                new_w, new_h = resized_augmented_img.shape[1], resized_augmented_img.shape[0]
                if x + new_w > image_size[0] - padding:
                    new_w = image_size[0] - padding - x
                    resized_augmented_img = cv2.resize(resized_augmented_img, (new_w, new_h))
                if y + new_h > image_size[1] - padding:
                    new_h = image_size[1] - padding - y
                    resized_augmented_img = cv2.resize(resized_augmented_img, (new_w, new_h))

                image[y:y+new_h, x:x+new_w] = resized_augmented_img[:new_h, :new_w]
                annotations.append({
                    "label": str(i),
                    "bbox": [x, y, x + new_w, y + new_h]
                })
                break
            attempts += 1
        if attempts == 100:
            raise RuntimeError("Could not place image without overlap after 100 attempts")
    
    image_name = f"generates_{image_index}.jpg"
    output_image_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_image_path, image)
    
    return {
        "file": image_name,
        "annotations": annotations
    }

def draw_annotations_for_all_images(annotations_json_path, images_dir):
    with open(annotations_json_path, 'r') as json_file:
        annotations_data = json.load(json_file)
    
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        image_annotations = next((item['annotations'] for item in annotations_data['images'] if item['file'] == image_name), [])
        
        image = cv2.imread(image_file)
        for annotation in image_annotations:
            x1, y1, x2, y2 = annotation['bbox']
            label = annotation['label']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Annotated Image', image)
        key = cv2.waitKey(0)
        if key == ord('n'):
            continue
        elif key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, "Targets")
    target_generate_dir = os.path.join(script_dir, "Targets_generates")
    output_dir = os.path.join(script_dir, "generates_images")
    backgrounds_dir = os.path.join(script_dir, "backgrounds")
    os.makedirs(output_dir, exist_ok=True)

    if DO_GENERATES == True:
        create_target_images_with_border(target_dir, target_generate_dir)
        
        annotations_list = []

        for i in range(1, GEN_CNT+1):
            annotation = generate_fhd_image_with_targets(i, target_generate_dir, output_dir, backgrounds_dir)
            annotations_list.append(annotation)
        
        json_output = {
            "images": annotations_list
        }
        
        with open(os.path.join(output_dir, "labels.json"), "w") as json_file:
            json.dump(json_output, json_file, indent=4)

        print(f"Generated images and labels JSON saved at {output_dir}")

    # Draw annotations for all images
    else:
        draw_annotations_for_all_images(os.path.join(output_dir, "labels.json"), output_dir)
