import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import io
import cv2
import os
import tempfile
import torch
# import torchvision
from torchvision import transforms#, datasets, models
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# define class name and color for video
class_names_all = {
    1: 'Bus',
    2: 'Car',
    3: 'Motorcycle',
    4: 'Pick Up Car',
    5: 'Truck',
    6: 'Truck Box'}
class_colors_all = {
    'Bus': (0, 255, 0),           # Green
    'Car': (0, 0, 255),           # Blue
    'Motorcycle': (255, 165, 0),  # Orange
    'Pick Up Car': (255, 0, 0),   # Red
    'Truck': (0, 255, 255),       # Cyan
    'Truck Box': (255, 0, 255)}    # Magenta

# plot img function
def plot_image_from_output_pred(img, pred, total_objects):
    img = img.squeeze(0).cpu().permute(1, 2, 0)
    
    class_names =  {
        1: 'Bus',
        2: 'Car',
        3: 'Motorcycle',
        4: 'Pick Up Car',
        5: 'Truck',
        6: 'Truck Box'
    }

    class_colors = {
        'Bus': 'green',
        'Car': 'blue',
        'Motorcycle': 'orange',
        'Pick Up Car': 'red',
        'Truck': 'cyan',
        'Truck Box': 'magenta',
    }
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Display the total count of objects detected in the image
    ax.text(5, 15, f"Total Detected Object: {total_objects}", color='white', fontsize=8, weight='bold',
            bbox={'facecolor': 'black', 'alpha': 1, 'pad': 3})

    for idx in range(len(pred["boxes"])):
        xmin, ymin, xmax, ymax = pred["boxes"][idx]
        class_label = pred['labels'][idx].item()
        class_name = class_names.get(class_label, 'Unknown')

        rect = patches.Rectangle(
            (xmin, ymin),
            (xmax - xmin),
            (ymax - ymin),
            linewidth=1,
            edgecolor=class_colors.get(class_name, 'black'),
            facecolor='none'
        )

        ax.add_patch(rect)

        # Add label text with scores
        scores = pred['scores'].cpu().numpy()
        score = scores[idx]
        label_text = f"{class_name}\n{score:.3f}"

        ax.text(xmin, ymin, label_text, color='white', fontsize=8, weight='bold', verticalalignment='top',
                bbox={'facecolor': class_colors.get(class_name, 'black'), 'alpha': 0.7, 'pad': 0})

    # Remove white background
    fig.patch.set_alpha(0)
    ax.set_alpha(0)

    # Convert the Matplotlib figure to an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_data = buf.getvalue()
    
    # Display the image using Streamlit
    st.image(img_data, caption='Predicted Objects', use_column_width=True)

    plt.close(fig)


# predict function
def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)):
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']):
            if score > threshold:
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list].cpu()
        preds[id]['labels'] = preds[id]['labels'][idx_list].cpu()
        preds[id]['scores'] = preds[id]['scores'][idx_list].cpu()
    return preds

# divide the display
col1, col2 = st.columns(2)

def pred_image(uploaded_file, pred_threshold, model):
    img = Image.open(uploaded_file)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = make_prediction(model, img_tensor, pred_threshold)

    # Count objects detected in the image
    total_objects = sum(len(preds[_idx]['labels']) for _idx in range(len(preds)))

    for _idx in range(len(preds)):
        prediction_labels = [class_names_all[label] for label in preds[_idx]['labels'].cpu().numpy()]
        prediction_scores = preds[_idx]['scores'].cpu().numpy()

        # Create a list of lists for the table
        table_data = [["Label", "Score"]]
        for label, score in zip(prediction_labels, prediction_scores):
            table_data.append([label, f"{score:.3f}"])
            
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df.index +=1
        df = df.rename_axis(None)

        # Display the table
        with col1:
            st.write(f"Total Predicted Object: {len(df)}")
            st.dataframe(df)
        with col2:
            plot_image_from_output_pred(img_tensor.cpu(), preds[_idx], total_objects)


def pred_vid_NRT(uploaded_file, pred_threshold, model, dir_path):

    video_file = st.video(uploaded_file)
    video_path = "video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        st.error('Error opening video file.')

    # Get the video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a loading message
    loading_message = st.empty()
    loading_message.text('Converting video with detected objects...')
    
    # Create a loading bar
    progress_bar = st.progress(0)

    # Process each frame of the video
    processed_frames = []
    for frame_index in range(num_frames):
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to PIL Image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # Apply transformations
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            preds = make_prediction(model, image, pred_threshold)

        # Count objects detected in this frame
        total_objects = 0

        # Draw bounding boxes on the frame
        for _idx in range(len(preds)):
            for idx in range(len(preds[_idx]['boxes'])):
                xmin, ymin, xmax, ymax = preds[_idx]['boxes'][idx].cpu().numpy()
                class_label = preds[_idx]['labels'][idx].item()
                class_name = class_names_all[class_label]
                class_color = class_colors_all[class_name]
                class_color_rgb = class_colors_all[class_name]

                total_objects += 1

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color_rgb, 2)
                
                # Add label text with scores
                score = preds[_idx]['scores'][idx].item()
                label_text = f"{class_name} {score:.3f}"
                cv2.putText(frame, label_text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color_rgb, 2)

        # Draw background rectangle for the text
        bg_rect_color = (0, 0, 0)
        text_bg_height = 25
        cv2.rectangle(frame, (5, 0), (width, text_bg_height), bg_rect_color, -1)

        # Display the total count of objects detected in this frame
        count_text = f"Total Predicted Object : {total_objects}"
        text_color = (255, 255, 255)
        cv2.putText(frame, count_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Append the processed frame to the list
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        processed_frames.append(frame)

        # Update the progress bar
        progress = (frame_index + 1) / num_frames
        progress_bar.progress(progress)

    video.release()

    # Create a temporary directory and generate a unique file name
    output_dir = dir_path#tempfile.mkdtemp()
    output_path = os.path.join(output_dir, "output.mp4")

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write the processed frames to the video file
    for frame in processed_frames:
        output_video.write(frame)

    # Release the video writer
    output_video.release()

    # Display a success message and a download link for the output video file
    st.success('Video conversion with detected objects completed!')
    
    st.video(output_path)
    
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="output.mp4")


def pred_vid_RT(uploaded_file, pred_threshold, model):
    video_file = st.video(uploaded_file)
    video_path = "video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        st.error('Error opening video file.')

    # Get the video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()

    # Process each frame of the video
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to PIL Image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        # Apply transformations
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)

        # Add batch dimension
        image = image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            preds = make_prediction(model, image, pred_threshold)

        # Count objects detected in this frame
        total_objects = 0

        # Draw bounding boxes on the frame and count objects
        for _idx in range(len(preds)):
            for idx in range(len(preds[_idx]['boxes'])):
                xmin, ymin, xmax, ymax = preds[_idx]['boxes'][idx].cpu().numpy()
                class_label = preds[_idx]['labels'][idx].item()
                class_name = class_names_all[class_label]
                class_color = class_colors_all[class_name]
                class_color_rgb = class_colors_all[class_name]

                total_objects += 1

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_color_rgb, 2)

                # Add label text with scores
                score = preds[_idx]['scores'][idx].item()
                label_text = f"{class_name} {score:.3f}"
                cv2.putText(frame, label_text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color_rgb, 2)

        # Draw background rectangle for the text
        bg_rect_color = (0, 0, 0)
        text_bg_height = 25
        cv2.rectangle(frame, (5, 0), (width, text_bg_height), bg_rect_color, -1)

        # Display the total count of objects detected in this frame
        count_text = f"Total Predicted Object : {total_objects}"
        text_color = (255, 255, 255)
        cv2.putText(frame, count_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Display the frame with bounding boxes and counts
        frame_placeholder.image(frame, channels='RGB')

    video.release()

