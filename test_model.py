import os
import cv2
from ultralytics import YOLO
import glob
import json
import numpy as np


def is_center_inside_box(center_x, center_y, x1, y1, x2, y2):
    return x1 <= center_x <= x2 and y1 <= center_y <= y2

def convert_bbox_to_resized_scale(bbox, orig_width, orig_height, new_width, new_height):
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * new_width / orig_width)
    y1 = int(y1 * new_height / orig_height)
    x2 = int(x2 * new_width / orig_width)
    y2 = int(y2 * new_height / orig_height)
    return x1, y1, x2, y2

def apply_custom_sharpening_filter(image):
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def apply_laplacian_sharpening(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened = cv2.convertScaleAbs(image - laplacian)
    return sharpened

def apply_unsharp_mask(image, kernel_size=(3, 3), sigma=1, amount=1):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)    
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)    
    return sharpened


def load_and_predict(image_path, model, width, height):
    image = cv2.imread(image_path)

    image = cv2.resize(image, (width, height))
     # cv2.imshow('before', image)
    # image = apply_custom_sharpening_filter(image)
    # cv2.imshow('after', image)
    results = model.predict(image, imgsz=[width, height], verbose=False)
    predictions = []
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            score = box.conf[0].cpu().item()
            if score < 0.5:
                continue
            label = int(box.cls[0].cpu().item())
            x1, y1, x2, y2 = map(int, coords)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            predictions.append((label, center_x, center_y, x1, y1, x2, y2))
    return predictions

def evaluate_model(dataset_dirs, model_path, orig_width, orig_height, new_width, new_height, auto_mode=True):
    print("test start")
    model = YOLO(model_path)
    correct = 0
    error = 0
    total = 0
    yolo_detect_cnt = 0
    yolo_miss_cnt = 0
    total_cnt = 0

    print("init done")
    for dataset_dir in dataset_dirs:
        dataset_dir = os.path.normpath(dataset_dir)
        image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))
        label_file = os.path.join(dataset_dir, "labels.json")

        if not os.path.exists(label_file):
            print(f"Label file {label_file} not found.")
            continue

        with open(label_file, 'r') as f:
            labels_data = json.load(f)

        for image_file in image_files:
            image_name = os.path.basename(image_file)
            predictions = load_and_predict(image_file, model, new_width, new_height)
            yolo_detect_cnt += len(predictions)  # Add YOLO detected count
            total_cnt += 10

            # Find corresponding annotations
            annotations = next((item['annotations'] for item in labels_data['images'] if item['file'] == image_name), [])

            if not annotations:  # Skip images without annotations
                continue

            draw_frame = cv2.imread(image_file)
            draw_frame = cv2.resize(draw_frame, (new_width, new_height))

            matched_annotations = set()

            # Check each prediction against all annotations
            for label, center_x, center_y, x1, y1, x2, y2 in predictions:
                total += 1
                match_found = False
                for annotation in annotations:
                    ann_label = int(annotation['label'])
                    ann_bbox = annotation['bbox']
                    ann_x1, ann_y1, ann_x2, ann_y2 = convert_bbox_to_resized_scale(ann_bbox, orig_width, orig_height, new_width, new_height)
                    if is_center_inside_box(center_x, center_y, ann_x1, ann_y1, ann_x2, ann_y2) and label == ann_label:
                        correct += 1
                        matched_annotations.add(tuple(annotation['bbox']))  # Convert annotation to hashable type
                        match_found = True
                        break
                if not match_found:
                    error += 1

                # Draw detection boxes and centers
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 파란색 박스
                cv2.circle(draw_frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)  # 빨간색 점
                cv2.putText(draw_frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)  # 파란색 숫자

            # Check for annotations that were not detected
            for annotation in annotations:
                if tuple(annotation['bbox']) not in matched_annotations:
                    yolo_miss_cnt += 1

            # Draw ground truth boxes
            for annotation in annotations:
                ann_bbox = annotation['bbox']
                ann_x1, ann_y1, ann_x2, ann_y2 = convert_bbox_to_resized_scale(ann_bbox, orig_width, orig_height, new_width, new_height)
                ann_label = int(annotation['label'])
                cv2.rectangle(draw_frame, (ann_x1, ann_y1), (ann_x2, ann_y2), (0, 255, 0), 1)  # 초록색 박스
                cv2.putText(draw_frame, str(ann_label), (ann_x1, ann_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # 초록색 숫자, 위치 조정

            # Display the image with annotations
            info_text = f"Total: {total}, Correct: {correct}, Error: {error}, YOLO Detect Count: {yolo_detect_cnt}, YOLO Miss Count: {yolo_miss_cnt} Total: {total_cnt}"
            cv2.putText(draw_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 결과를 좌측 상단에 표시

            # cv2.imshow('Detection and Ground Truth', draw_frame)
            if auto_mode:
                cv2.waitKey(1)
            else:
                while True:
                    key = cv2.waitKey(0)
                    if key == ord('n'):
                        break
                    elif key == ord('q'):
                        cv2.destroyAllWindows()
                        return correct, error, total, yolo_detect_cnt, yolo_miss_cnt, total_cnt

    cv2.destroyAllWindows()
    return correct, error, total, yolo_detect_cnt, yolo_miss_cnt, total_cnt

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dirs = [
        os.path.join(script_dir, "Sample_1m_1st"),
        os.path.join(script_dir, "Sample_1m_2nd"),
        os.path.join(script_dir, "images/capture"),
        os.path.join(script_dir, "images/image"),
        # 추가 데이터셋 폴더 경로를 여기에 추가
    ]
    # model_path = os.path.join(script_dir, "generates_images", "best.pt")
    model_path = os.path.join(script_dir, "../FINAL\devDreamers\LGClientDisplayPyQT\image_algo\models", "best_1.pt")
    orig_width, orig_height = 1920, 1080
    new_width, new_height = 960, 544

    auto_mode = True  # 자동 모드 또는 수동 모드 선택

    correct, error, total, yolo_detect_cnt, yolo_miss_cnt, total_cnt = evaluate_model(dataset_dirs, model_path, orig_width, orig_height, new_width, new_height, auto_mode)


    # YOLO 탐지 비율 계산
    if total > 0:
        yolo_detect_rate = (correct / total) * 100
    else:
        yolo_detect_rate = 0

    print(f"Total: {total}, Correct: {correct}, Error: {error}, YOLO Detect Count: {yolo_detect_cnt}, YOLO Miss Count: {yolo_miss_cnt}")
    print(f"YOLO Detect Rate: {yolo_detect_rate:.2f}%")
