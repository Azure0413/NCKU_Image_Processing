from ultralytics import YOLO
import os
import json
import supervision as sv
import cv2
import numpy as np
from tkinter import filedialog, messagebox
import shutil
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

# 定義 Load Folder 功能
def load_folder():
    # Open file dialog to select a folder
    folder_path = filedialog.askdirectory(title="Select Data Folder")

    if not folder_path:
        return  # If no folder selected, do nothing

    folder_name = os.path.basename(folder_path)

    # Validate the folder name is "data"
    if folder_name != "data":
        messagebox.showerror("Invalid Folder", "The folder name must be 'data'.")
        return

    # Check for required subfolders
    scaphoid_images_path = os.path.join(folder_path, "scaphoid_detection", "images")
    scaphoid_annotations_path = os.path.join(folder_path, "scaphoid_detection", "annotations")
    fracture_annotations_path = os.path.join(folder_path, "fracture_detection", "annotations")

    if not (os.path.exists(scaphoid_images_path) and 
            os.path.exists(scaphoid_annotations_path) and 
            os.path.exists(fracture_annotations_path)):
        messagebox.showerror(
            "Invalid Folder Structure", 
            "The 'data' folder must contain the following subfolders:\n"
            "- ./scaphoid_detection/images\n"
            "- ./scaphoid_detection/annotations\n"
            "- ./fracture_detection/annotations"
        )
        return

    # Create temp folder if it doesn't exist
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # Copy the selected folder to the temp directory
    dest_folder = os.path.join(temp_folder, "data")
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)  # Remove existing folder to avoid conflicts

    shutil.copytree(folder_path, dest_folder)

    # Show success message
    messagebox.showinfo("Success", f"Folder successfully uploaded to '{temp_folder}/data'.")

def preprocess_coordinary():
    images_folder = "./temp/data/scaphoid_detection/images"
    annotations_folder = "./temp/data/scaphoid_detection/annotations"
    annotations_fracture_folder = "./temp/data/fracture_detection/annotations"
    output_data_folder = "./temp/data"

    output_annotations_folder = os.path.join(output_data_folder, "annotations")
    output_images_folder = os.path.join(output_data_folder, "images")
    output_scaphoid_annotations_folder = os.path.join(output_data_folder, "scaphoid_annotations")

    os.makedirs(output_annotations_folder, exist_ok=True)
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_scaphoid_annotations_folder, exist_ok=True)

    annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith('.json')]
    fracture_files = [f for f in os.listdir(annotations_fracture_folder) if f.endswith('.json')]

    for annotation_file in annotation_files:
        image_file = annotation_file.replace('.json', '.jpg')
        image_path = os.path.join(images_folder, image_file)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_file} not found in {images_folder}, skipping...")
            continue

        # 读取对应的 annotations 和 annotations_fracture 文件
        annotation_path = os.path.join(annotations_folder, annotation_file)
        fracture_path = os.path.join(annotations_fracture_folder, annotation_file)

        if not os.path.exists(fracture_path):
            print(f"Warning: Fracture data {annotation_file} not found, skipping...")
            continue

        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        with open(fracture_path, 'r') as f:
            fracture_data = json.load(f)

        if len(annotation_data) == 0 or "bbox" not in annotation_data[0]:
            print(f"Warning: No bbox in {annotation_file}, writing null annotation...")
            output_data = [{"name": None, "bbox": None}]
        else:
            bbox = annotation_data[0]["bbox"]
            x1, y1 = int(bbox[0]), int(bbox[1])

            # 获取 annotations_fracture 的四点坐标
            if len(fracture_data) == 0 or fracture_data[0]["name"] is None:
                output_data = [{"name": None, "bbox": None}]
            else:
                fracture_bbox = fracture_data[0]["bbox"]
                if fracture_bbox is None:
                    output_data = [{"name": None, "bbox": None}]
                else:
                    # 偏移 fracture_bbox 坐标
                    offset_bbox = []
                    for point in fracture_bbox:
                        offset_bbox.append([point[0] + x1, point[1] + y1])
                    output_data = [{"name": "Fracture", "bbox": offset_bbox}]

        # 保存到输出文件夹
        output_path = os.path.join(output_annotations_folder, annotation_file)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

    # 複製原始圖像文件夾到輸出資料夾
    shutil.copytree(images_folder, output_images_folder, dirs_exist_ok=True)

    # 複製 scaphoid_annotations 文件夹到輸出資料夾
    shutil.copytree(annotations_folder, output_scaphoid_annotations_folder, dirs_exist_ok=True)

    # 刪除多餘資料夾，只保留指定的三個資料夾
    for item in os.listdir(output_data_folder):
        item_path = os.path.join(output_data_folder, item)
        if item not in ["annotations", "images", "scaphoid_annotations"]:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

def scaphoid_detection():
    # 加載訓練好的 YOLO 模型
    # model = YOLO('./scaphoid_detection_model/best.pt')
    model = YOLO('./scaphoid_detection_model/best_300.pt')

    # 設定資料夾路徑
    images_folder = "./temp/data/images"
    predicted_folder = "./temp/output_1/predicted"
    cropped_folder = "./temp/output_1/cropped"
    origin_file_path = "./temp/output_1/origin.txt"

    # 創建輸出資料夾
    os.makedirs(predicted_folder, exist_ok=True)
    os.makedirs(cropped_folder, exist_ok=True)

    # 打開檔案以寫入Bounding Box的座標
    with open(origin_file_path, "w") as origin_file:
        for image_name in os.listdir(images_folder):
            image_path = os.path.join(images_folder, image_name)

            # 讀取圖片
            image = cv2.imread(image_path)

            # 使用模型預測
            results = model(image_path)

            if results[0].boxes:
                # 選擇最高信心度的預測框
                max_confidence_idx = np.argmax(results[0].boxes.conf.tolist())
                pred_box = results[0].boxes.xyxy[max_confidence_idx].tolist()
                x1, y1, x2, y2 = map(int, pred_box)

                # 儲存裁切的Bounding Box
                cropped_image = image[y1:y2, x1:x2]
                cropped_image_path = os.path.join(cropped_folder, f"{image_name.split('.')[0]}.jpg")
                cv2.imwrite(cropped_image_path, cropped_image)

                # 在圖片上繪製預測的Bounding Box
                annotated_image = image.copy()
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 藍色框

                # 儲存繪製結果
                predicted_image_path = os.path.join(predicted_folder, image_name)
                cv2.imwrite(predicted_image_path, annotated_image)

                # 將Bounding Box的左上角座標寫入檔案
                origin_file.write(f"{image_name}: x1={x1}, y1={y1}\n")

def scaphoid_detection_iou():
    # 加載訓練好的 YOLO 模型
    # model = YOLO('./scaphoid_detection_model/best.pt')
    model = YOLO('./scaphoid_detection_model/best_300.pt')

    # 設定資料夾路徑
    images_folder = "./temp/data/images"
    annotations_folder = "./temp/data/scaphoid_annotations"

    def calculate_iou(box1, box2):
        """
        計算兩個 bounding box 的 IOU 值
        :param box1: [x1, y1, x2, y2]
        :param box2: [x1, y1, x2, y2]
        :return: IOU 值
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = box1_area + box2_area - intersection

        if union == 0:
            return 0

        return intersection / union

    total_iou = 0
    num_boxes = 0

    # 處理每張圖片
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        annotation_path = os.path.join(annotations_folder, image_name.replace('.jpg', '.json'))

        # 讀取圖片
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 讀取 Ground Truth 標註
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        ground_truth_boxes = [
            [int(bbox["bbox"][0]), int(bbox["bbox"][1]), int(bbox["bbox"][2]), int(bbox["bbox"][3])]
            for bbox in annotations
        ]

        # 使用 YOLO 模型進行預測
        results = model(image_path)

        # 取得 confidence 最大的預測框
        if results[0].boxes:
            max_confidence_idx = np.argmax(results[0].boxes.conf.tolist())
            pred_box = results[0].boxes.xyxy[max_confidence_idx].tolist()
            x1, y1, x2, y2 = map(int, pred_box)

            # 在圖片上繪製 Ground Truth 和預測的 Bounding Box
            annotated_image = image.copy()

            # 畫出 Ground Truth
            for gt_box in ground_truth_boxes:
                cv2.rectangle(annotated_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 2)  # 綠色框

            # 畫出預測框並計算 IOU
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 藍色框

            for gt_box in ground_truth_boxes:
                iou = calculate_iou([x1, y1, x2, y2], gt_box)
                total_iou += iou
                num_boxes += 1

    # 計算並輸出 Mean IOU
    mean_iou = total_iou / num_boxes if num_boxes > 0 else 0
    return mean_iou

def fracture_detection_output():
    # 加載訓練好的 YOLO 模型
    # model = YOLO('./fracture_detection_model/best.pt')
    # model = YOLO('./fracture_detection_model/augmented_train_new.pt')
    # model = YOLO('./fracture_detection_model/all_data.pt')
    model = YOLO('./fracture_detection_model/best_final.pt')
    # model = YOLO('./fracture_detection_model/all_data_aug.pt')
    # model = YOLO('./fracture_detection_model/best_300.pt')
    # model = YOLO('./fracture_detection_model/AP0.pt')

    # 測試影像的資料夾
    test_images_folder = "./temp/output_1/cropped"
    output_images_folder = "./temp/output_2/small_output"

    # 確保輸出資料夾存在
    os.makedirs(output_images_folder, exist_ok=True)

    # 讀取測試影像檔案
    image_files = sorted([f for f in os.listdir(test_images_folder) if f.endswith('.jpg')])

    # 處理每張影像
    for img_file in image_files:
        file_path = os.path.join(test_images_folder, img_file)

        # 使用模型進行推論
        results = model(file_path)

        # 轉換為 Supervision 格式的檢測結果
        detections = sv.Detections.from_ultralytics(results[0])

        # 初始化 OrientedBoxAnnotator
        oriented_box_annotator = sv.OrientedBoxAnnotator()

        # 讀取圖片並繪製檢測框
        frame = cv2.imread(file_path)
        annotated_frame = oriented_box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # 保存繪製好的影像
        output_path = os.path.join(output_images_folder, img_file)
        cv2.imwrite(output_path, annotated_frame)


def fracture_detection_acc_p_r():
    # 加載訓練好的 YOLO 模型
    # model = YOLO('./fracture_detection_model/best.pt')
    # model = YOLO('./fracture_detection_model/augmented_train_new.pt')
    # model = YOLO('./fracture_detection_model/all_data.pt')
    model = YOLO('./fracture_detection_model/best_final.pt')
    # model = YOLO('./fracture_detection_model/all_data_aug.pt')
    # model = YOLO('./fracture_detection_model/best_300.pt')
    # model = YOLO('./fracture_detection_model/AP0.pt')

    # 資料夾路徑
    test_images_folder = "./temp/output_1/cropped"
    annotations_folder = "./temp/data/annotations"

    # 儲存預測結果
    predictions = []

    # 讀取測試影像檔案
    image_files = sorted([f for f in os.listdir(test_images_folder) if f.endswith('.jpg')])

    # 處理每張影像
    for img_file in image_files:
        file_path = os.path.join(test_images_folder, img_file)

        # 使用模型進行推論
        results = model(file_path)

        # 轉換為 Supervision 格式的檢測結果
        detections = sv.Detections.from_ultralytics(results[0])

        # 判斷是否有 bounding box
        has_bounding_box = 1 if detections.xyxy.shape[0] > 0 else 0
        predictions.append((img_file, has_bounding_box))

    tp = fp = fn = tn = 0

    for img_file, pred_class_id in predictions:
        # Ground Truth
        annotation_path = os.path.join(annotations_folder, img_file.replace('.jpg', '.json'))
        ground_truth_class = 0  # 預設為 0 (無 Fracture)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)

            # 檢查是否有 "Fracture"
            for annotation in annotations:
                if annotation.get("name") == "Fracture":
                    ground_truth_class = 1
                    break

        # 計算 TP, FP, FN, TN
        if pred_class_id == 1 and ground_truth_class == 1:
            tp += 1
        elif pred_class_id == 1 and ground_truth_class == 0:
            fp += 1
        elif pred_class_id == 0 and ground_truth_class == 1:
            fn += 1
        elif pred_class_id == 0 and ground_truth_class == 0:
            tn += 1

    # 計算指標
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall

def final():
    # model = YOLO('./fracture_detection_model/best.pt')
    # model = YOLO('./fracture_detection_model/augmented_train_new.pt')
    # model = YOLO('./fracture_detection_model/all_data.pt') 
    model = YOLO('./fracture_detection_model/best_final.pt')
    # model = YOLO('./fracture_detection_model/all_data_aug.pt')
    # model = YOLO('./fracture_detection_model/best_300.pt')
    # model = YOLO('./fracture_detection_model/AP0.pt')

    # 路徑設置
    images_folder = "./temp/data/images"
    small_output_folder = "./temp/output_1/cropped"
    annotations_folder = "./temp/data/annotations"
    scaphoid_annotations_folder = "./temp/data/scaphoid_annotations"
    completed_output_folder = "./temp/output_2/completed_output"
    origin_txt_file = "./temp/output_1/origin.txt"

    # 創建輸出文件夾
    os.makedirs(completed_output_folder, exist_ok=True)

    # 讀取 origin.txt 文件，存儲偏移值
    offsets = {}
    with open(origin_txt_file, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            img_file = parts[0].strip()
            coords = parts[1].strip()
            coords_dict = {kv.split("=")[0].strip(): int(kv.split("=")[1].strip()) for kv in coords.split(", ")}
            offsets[img_file] = (coords_dict["x1"], coords_dict["y1"])

    mean_iou_list = []

    # 定義 IoU 計算函數（基於四邊形）
    def calculate_polygon_iou(boxA, boxB):
        polyA = Polygon(boxA)
        polyB = Polygon(boxB)

        if not polyA.is_valid or not polyB.is_valid:
            return 0

        intersection_area = polyA.intersection(polyB).area
        union_area = polyA.union(polyB).area

        return intersection_area / union_area if union_area > 0 else 0

    # 處理每張圖片
    for img_file, (offset_x, offset_y) in offsets.items():
        img_path = os.path.join(images_folder, img_file)
        small_output_path = os.path.join(small_output_folder, img_file)
        annotation_path = os.path.join(annotations_folder, f"{os.path.splitext(img_file)[0]}.json")
        scaphoid_annotation_path = os.path.join(scaphoid_annotations_folder, f"{os.path.splitext(img_file)[0]}.json")

        # 確認文件存在
        if not os.path.exists(img_path):
            print(f"Warning: {img_file} not found in {images_folder}, skipping...")
            continue
        if not os.path.exists(small_output_path):
            print(f"Warning: {img_file} not found in {small_output_folder}, skipping...")
            continue
        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation for {img_file} not found, skipping...")
            continue
        if not os.path.exists(scaphoid_annotation_path):
            print(f"Warning: Scaphoid annotation for {img_file} not found, skipping...")
            continue

        # 加載圖片
        image = cv2.imread(img_path)

        # 讀取 Ground Truth BBox
        with open(annotation_path, "r") as f:
            gt_data = json.load(f)

        # 檢查 JSON 結構
        if isinstance(gt_data, list):
            annotations = gt_data
        elif isinstance(gt_data, dict) and "annotations" in gt_data:
            annotations = gt_data["annotations"]
        else:
            print(f"Invalid JSON format in {annotation_path}, skipping...")
            continue

        gt_bboxes = []
        for obj in annotations:
            if obj.get("name") is None or obj.get("bbox") is None:
                continue
            bbox = obj["bbox"]
            if len(bbox) != 4:
                print(f"Invalid bbox format for object {obj}, skipping...")
                continue
            gt_bboxes.append(np.array(bbox, dtype=float))

        # 讀取 Scaphoid Annotation
        with open(scaphoid_annotation_path, "r") as f:
            scaphoid_data = json.load(f)

        # 繪製 Scaphoid Bounding Box
        for obj in scaphoid_data:
            if obj.get("name") != "Scaphoid" or obj.get("bbox") is None:
                continue
            bbox = obj["bbox"]
            if len(bbox) != 4:
                print(f"Invalid Scaphoid bbox format for object {obj}, skipping...")
                continue

            # 將 bbox 畫到圖片上
            top_left = (int(bbox[0]), int(bbox[1]))
            bottom_right = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(image, top_left, bottom_right, (255, 0, 255), 2)  # 紫色框

        # 推理模型
        results = model(small_output_path)
        for result in results:
            detections = sv.Detections.from_ultralytics(result)

            if "xyxyxyxy" not in detections.data:
                print(f"No bounding box data for {img_file}, skipping...")
                continue

            for bbox in detections.data["xyxyxyxy"]:
                bbox = np.array(bbox, dtype=float).flatten()
                bbox_offset = np.array(bbox, dtype=float) + np.tile([offset_x, offset_y], 4)
                polygon_points = bbox_offset.reshape((-1, 2))
                cv2.drawContours(image, [np.int0(polygon_points)], -1, (255, 0, 0), 2)  # 藍色框

            for pred_bbox in detections.data["xyxyxyxy"]:
                pred_bbox = np.array(pred_bbox, dtype=float).flatten()
                pred_polygon = np.array(pred_bbox, dtype=float) + np.tile([offset_x, offset_y], 4)
                pred_polygon = pred_polygon.reshape((-1, 2))
                iou_scores = [calculate_polygon_iou(pred_polygon, gt_bbox) for gt_bbox in gt_bboxes]
                if iou_scores:
                    mean_iou_list.append(max(iou_scores))

        for gt_box in gt_bboxes:
            cv2.drawContours(image, [np.int0(gt_box)], -1, (0, 0, 255), 2)  # 紅色框

        output_path = os.path.join(completed_output_folder, img_file)
        cv2.imwrite(output_path, image)

    mean_iou = sum(mean_iou_list) / len(mean_iou_list) if mean_iou_list else 0
    return mean_iou

# Function to handle analysis
def run_analysis():
    try:
        scaphoid_detection()
        mean_iou = scaphoid_detection_iou()
        fracture_detection_output()
        acc, prec, rec = fracture_detection_acc_p_r()
        iou = final()
        messagebox.showinfo("Analysis Complete", "Analysis has been successfully completed.")
        return mean_iou, iou, acc, prec, rec
    except Exception as e:
        messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")

def download_completed_output():
    # 定義要壓縮的目標資料夾
    target_folder = "./temp/output_2/completed_output"
    
    if not os.path.exists(target_folder):
        messagebox.showerror("Error", "The target folder does not exist.")
        return

    # 壓縮目標資料夾
    zip_file_path = "./completed_output.zip"
    shutil.make_archive(base_name="completed_output", format="zip", root_dir=target_folder)

    # 彈出保存文件的對話框
    save_path = filedialog.asksaveasfilename(
        defaultextension=".zip",
        filetypes=[("ZIP files", "*.zip")],
        initialfile="completed_output.zip",
        title="Save Completed Output"
    )
    if save_path:
        try:
            shutil.move(zip_file_path, save_path)
            messagebox.showinfo("Download", "The completed output has been downloaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save the file: {e}")
    else:
        # 如果用戶取消保存，刪除臨時壓縮文件
        os.remove(zip_file_path)
