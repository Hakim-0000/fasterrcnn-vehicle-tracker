#############################################################
"""
    This web app was built by me [(Luqman Hakim)](https://github.com/Hakim-0000)
    using streamlit and is used as one of the report for my University's final project
    to get graduated.
        
    If by any chance you find this apps helpful and want to learn more
    about it or you want to work with me, please contact me through email
    at luqmanhakim047300@gmail.com
"""


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
import torchvision
from torchvision import transforms, datasets, models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from load_trainedModel import load_models



home_dir = os.getcwd()

# @st.cache_resource
st.title('Vehicle in Traffic Detection Web App')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

with st.expander("short readme"):
    st.markdown("""
        This web app was built for detecting **Vehicle on a Traffic**, so if your
        images or video isn't related to ```Traffic```, you might get bad results or no results.
        
        This web apps might be slow because it is using the ```cpu``` as device.
        
        Short demo:
    """)
    st.image('assets/gif-contoh.gif')


det_mode = st.selectbox('Select how you want to detect objects in your file',
                        ('Real-time', 'Not Real-time'))
uploaded_file = st.file_uploader('File Upload', type=['png', 'jpg', 'jpeg', 'mp4'])

# sidebar configuration
with st.sidebar:
    if uploaded_file is not None:
        trained_model_versions = st.selectbox('Select the trained model version',
                                             ('v01', 'v02')
                                             )
        backbone = st.selectbox(
            'Select your backbone model',
            ('ResNet50_fpn', 'ResNet50_fpn_v2',
             'MobileNet_v3_large_fpn', 'MobileNet_v3_large_320_fpn'))
        pred_threshold = st.slider('Threshold score',
                                   0.0, 1.0, 0.5,
                                   disabled=False)
            
    else:
        st.write('**Please upload your file first!**')
        backbone = st.selectbox(
            'Select how you want to detect objects in your file',
            ('No mode selected', ''),
            disabled=True)
        
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

def pred_image(model):
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

def pred_vid_NRT(model):

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
    output_dir = tempfile.mkdtemp()
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
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="output.mp4")

    st.video(output_path)

def pred_vid_RT(model):
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


# divide the display
col1, col2 = st.columns(2)

def not_realtime():
    # NRT-D using ResNet50 v1
    if backbone == 'ResNet50_fpn': 
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_NRT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
    
    # NRT-D using ResNet50 v2
    elif backbone == 'ResNet50_fpn_v2':
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_NRT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
        
    # NRT-D using MobileNet v3 large
    elif backbone == 'MobileNet_v3_large_fpn':
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_NRT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
    
    # NRT-D using MobileNet v3 large 320
    elif backbone == 'MobileNet_v3_large_320_fpn':
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_NRT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
    
    else:
        # st.write('**Please choose your backbone!**')
        st.write('')

def realtime():
    # RT-D using ResNet50 v1
    if backbone == 'ResNet50_fpn': 
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_RT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
    
    # RT-D using ResNet50 v2
    elif backbone == 'ResNet50_fpn_v2':
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_RT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
        
    # RT-D using MobileNet v3 large
    elif backbone == 'MobileNet_v3_large_fpn':
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_RT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
    
    # NRT-D using MobileNet v3 large 320
    elif backbone == 'MobileNet_v3_large_320_fpn':
        file_ext = uploaded_file.name.split(".")[-1].lower()
        # Get base model and load it with the trained weight
        # model_used = get_trained_model(backbone, device)
        model_used = load_models(trained_model_versions, backbone, device)
        
        if file_ext in ['png', 'jpg', 'jpeg']:
            pred_image(model=model_used)
                    
        elif file_ext == 'mp4':
            pred_vid_RT(model=model_used)
        
        else:
            st.write('please use `.png`, `.jpg`, `.jpeg`, or `.mp4` extension instead!')
    
    else:
        # st.write('**Please choose your backbone!**')
        st.write('')


def main():
    if det_mode == 'Real-time':
        realtime() 
    
    # Real Time Detection
    elif det_mode == 'Not Real-time':
        not_realtime()


if __name__ == "__main__":
    main()
