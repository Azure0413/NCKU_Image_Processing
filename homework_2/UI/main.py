import os
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from utils import load_folder, run_analysis, download_completed_output, preprocess_coordinary

# Initialize main window
root = tk.Tk()
root.title("Image Processing Application")
root.geometry("1725x800")

# Global variables for image navigation
current_image_index = 0
image_list = []

def display_image():
    global current_image_index, image_list

    if not image_list:
        current_image_label.config(text="No images to display.")
        return

    image_name = image_list[current_image_index]
    image_path = os.path.join("temp", "data", "images", image_name)
    current_image_label.config(text=f"Current Image: {image_name}")

    # 顯示原始圖片
    try:
        img = Image.open(image_path)
        original_width, original_height = img.size
        new_width = original_width // 4
        new_height = original_height // 4
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        current_image_canvas.image = tk_img
        current_image_canvas.config(width=new_width, height=new_height)
        current_image_canvas.create_image(0, 0, anchor="nw", image=tk_img)
    except Exception as e:
        current_image_label.config(text=f"Error loading image: {e}")

    # 顯示裁剪後的圖片
    cropped_image_path = os.path.join("temp", "output_1", "cropped", f"{image_name.split('.')[0]}.jpg")
    try:
        cropped_img = Image.open(cropped_image_path)
        cropped_width, cropped_height = cropped_img.size
        new_cropped_width = cropped_width
        new_cropped_height = cropped_height
        cropped_img = cropped_img.resize((new_cropped_width, new_cropped_height), Image.Resampling.LANCZOS)
        tk_cropped_img = ImageTk.PhotoImage(cropped_img)
        segmented_image_canvas.image = tk_cropped_img
        segmented_image_canvas.config(width=new_cropped_width, height=new_cropped_height)
        segmented_image_canvas.create_image(0, 0, anchor="nw", image=tk_cropped_img)
    except Exception as e:
        segmented_image_label.config(text=f"No segmented image available for: {image_name}")

    # 顯示fracture detection圖片
    fracture_image_path = os.path.join("temp", "output_2", "small_output", f"{image_name.split('.')[0]}.jpg")
    try:
        fracture_img = Image.open(fracture_image_path)
        fracture_width, fracture_height = fracture_img.size
        fracture_img = fracture_img.resize((fracture_width, fracture_height), Image.Resampling.LANCZOS)
        tk_fracture_img = ImageTk.PhotoImage(fracture_img)
        fracture_detection_canvas.image = tk_fracture_img
        fracture_detection_canvas.config(width=fracture_width, height=fracture_height)
        fracture_detection_canvas.create_image(0, 0, anchor="nw", image=tk_fracture_img)
    except Exception as e:
        fracture_detection_label.config(text=f"No fracture detection image available for: {image_name}")
    
    # 顯示fracture detection圖片
    fracture_image_completed_path = os.path.join("temp", "output_2", "completed_output", f"{image_name.split('.')[0]}.jpg")
    try:
        fracture_image_completed = Image.open(fracture_image_completed_path)
        fracture_c_width, fracture_c_height = fracture_image_completed.size
        fracture_image_completed = fracture_image_completed.resize((fracture_c_width // 4, fracture_c_height // 4), Image.Resampling.LANCZOS)
        tk_fracture_img_completed = ImageTk.PhotoImage(fracture_image_completed)
        fracture_detection_completed_canvas.image = tk_fracture_img_completed
        fracture_detection_completed_canvas.config(width=fracture_c_width // 4 , height=fracture_c_height // 4)
        fracture_detection_completed_canvas.create_image(0, 0, anchor="nw", image=tk_fracture_img_completed)
    except Exception as e:
        fracture_detection_label.config(text=f"No fracture detection completed image available for: {image_name}")

# 下一张图片
def next_image():
    global current_image_index
    if image_list and current_image_index < len(image_list) - 1:
        current_image_index += 1
        display_image()

# 上一张图片
def previous_image():
    global current_image_index
    if image_list and current_image_index > 0:
        current_image_index -= 1
        display_image()

def clear_temp():
    global current_image_index, image_list

    # 清空 temp 資料夾
    temp_folder = "temp"
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)  # 刪除 temp 資料夾及其內容

    # 重新創建空的 temp 資料夾
    os.makedirs(temp_folder)

    # 重置全域變數和 UI 元件
    current_image_index = 0
    image_list = []

    # 清除 UI 中的圖片
    current_image_label.config(text="Current Image:")
    current_image_canvas.delete("all")
    segmented_image_label.config(text="Segmented Image:")
    segmented_image_canvas.delete("all")
    fracture_detection_label.config(text="Fracture Detection:")
    fracture_detection_canvas.delete("all")
    fracture_detection_completed_label.config(text="Fracture Detection Completed:")
    fracture_detection_completed_canvas.delete("all")

    # 清除所有評估數值
    # mean_iou_value.config(text="")
    for metric in ["iouvalue", "accuracyvalue", "precisionvalue", "recallvalue"]:
        metric_widget = detection_frame.children.get(metric)
        if metric_widget:
            metric_widget.config(text="")

    # 彈出提示訊息
    messagebox.showinfo("Clear", "The system has been reset.")

def analyze_and_update():
    global image_list

    # Run analysis logic
    mean_iou, iou, acc, prec, rec = run_analysis()

    # Update image list
    image_folder = os.path.join("temp", "data", "images")
    if os.path.exists(image_folder):
        image_list = sorted(os.listdir(image_folder))
        if image_list:
            display_image()
        else:
            messagebox.showinfo("Info", "No images found after analysis.")
    else:
        messagebox.showinfo("Info", "Analysis completed but no images found.")

    evaluation_values = {"Mean IOU": mean_iou, "IOU": iou, "Accuracy": acc, "Precision": prec, "Recall": rec}
    for metric, value in evaluation_values.items():
        if metric == 'Mean IOU':
            widget_name = f"{metric.lower().replace(' ', '')}value"  # 生成對應的name屬性名稱
            metric_value_widget = segmentation_frame.children.get(widget_name)
        else:
            widget_name = f"{metric.lower().replace(' ', '')}value"  # 生成對應的name屬性名稱
            metric_value_widget = detection_frame.children.get(widget_name)
        if metric_value_widget:  # 如果找到對應的Label，更新其文本內容
            metric_value_widget.config(text=f"{value:.4f}")

# Main frame to hold all blocks horizontally
main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Data Block
data_frame = ttk.LabelFrame(main_frame, text="Data", width=350)
data_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n")
data_frame.grid_propagate(False)

btn_frame = ttk.Frame(data_frame)
btn_frame.pack(pady=5)

load_button = tk.Button(btn_frame, text="Load Folder", command=load_folder)
load_button.grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Preprocessing", command=preprocess_coordinary).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Analysis", command=analyze_and_update).grid(row=0, column=2, padx=5)

current_image_label = ttk.Label(data_frame, text="Current Image:")
current_image_label.pack(anchor="w", pady=5)

current_image_canvas = tk.Canvas(data_frame, width=400, height=200, bg="white")
current_image_canvas.pack(pady=5)

navigation_frame = ttk.Frame(data_frame)
navigation_frame.pack(pady=5)

tk.Button(navigation_frame, text="Previous", command=previous_image).pack(side="left", padx=5)
tk.Button(navigation_frame, text="Next", command=next_image).pack(side="left", padx=5)

# Segmentation Block
segmentation_frame = ttk.LabelFrame(main_frame, text="Segmentation", width=350)
segmentation_frame.grid(row=0, column=1, padx=10, pady=5, sticky="n")
segmentation_frame.grid_propagate(False)

segmented_image_label = ttk.Label(segmentation_frame, text="Segmented Image:")
segmented_image_label.pack(anchor="w", pady=5)

segmented_image_canvas = tk.Canvas(segmentation_frame, width=400, height=200, bg="white")
segmented_image_canvas.pack(pady=5)

mean_iou_label = ttk.Label(segmentation_frame, text="Mean IOU:")
mean_iou_label.pack(anchor="w", pady=5)
meaniou_name = 'Mean IOU'
mean_iou_value = ttk.Label(segmentation_frame, text="", relief=tk.SUNKEN, width=20, name=f"{meaniou_name.lower().replace(' ', '')}value")
mean_iou_value.pack(pady=5)

# Detection Block
detection_frame = ttk.LabelFrame(main_frame, text="Detection", width=800)
detection_frame.grid(row=0, column=2, padx=10, pady=5, sticky="n")
detection_frame.grid_propagate(False)

# 左側區域（包含 Fracture Detection）
left_frame = ttk.Frame(detection_frame, width=400, height=400)
left_frame.pack(side="left", padx=10, pady=5, fill="y", anchor="n")
left_frame.pack_propagate(False)

# 左上 - Fracture Detection
fracture_detection_label = ttk.Label(left_frame, text="Fracture Detection:")
fracture_detection_label.pack(anchor="n", pady=5) 

fracture_detection_canvas = tk.Canvas(left_frame, width=400, height=200, bg="white")
fracture_detection_canvas.pack(anchor="n", pady=5) 

# 右側區域（Evaluation 指標）
detection_frame = ttk.Frame(detection_frame)
detection_frame.pack(side="left", padx=10, pady=5)

fracture_detection_completed_label = ttk.Label(detection_frame, text="Fracture Detection Completed:")
fracture_detection_completed_label.pack(anchor="w", pady=5)

fracture_detection_completed_canvas = tk.Canvas(detection_frame, width=400, height=200, bg="white")
fracture_detection_completed_canvas.pack(pady=5)

evaluation_label = ttk.Label(detection_frame, text="Evaluation:")
evaluation_label.pack(anchor="w", pady=5)

for metric in ["IOU", "Accuracy", "Precision", "Recall"]:
    metric_label = ttk.Label(detection_frame, text=f"{metric}:")
    metric_label.pack(anchor="w", pady=2)
    metric_value = ttk.Label(detection_frame, text="", relief=tk.SUNKEN, width=20, name=f"{metric.lower()}value")
    metric_value.pack(pady=2)


# Clear and Download Buttons
bottom_frame = ttk.Frame(root)
bottom_frame.pack(pady=10)

tk.Button(bottom_frame, text="Clear", command=clear_temp).pack(side="left", padx=5)
download_button = tk.Button(bottom_frame, text="Download", command=download_completed_output)
download_button.pack(side="left", padx=5)

# Run the application
root.mainloop()
