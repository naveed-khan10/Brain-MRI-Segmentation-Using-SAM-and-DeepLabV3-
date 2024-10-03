import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
import numpy as np

# Custom DeepLabV3 Model Class
class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1):
        super(DeepLabV3, self).__init__()
        # Load the pre-trained DeepLabV3 model
        self.model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
        # Modify the classifier for your number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)

# Set the device to CPU
device = torch.device("cpu")

# Initialize the model and move it to the CPU
model = DeepLabV3(num_classes=1)
model.to(device)

# Load the model weights onto the CPU
try:
    model.load_state_dict(torch.load('model_1.pth', map_location=device))
    model.eval()  # Set the model to evaluation mode
    st.success("Model loaded successfully on CPU!")
except Exception as e:
    st.error(f"Error loading model: {e}")

def preprocess_image(image):
    """ Preprocess the uploaded TIFF image to the format expected by the model """
    # Convert the image to RGB
    image = image.convert('RGB')
    
    # Define the preprocessing transformations
    transform = T.Compose([
        T.Resize((256, 256)),  # Resize to match the model's expected input size
        T.ToTensor(),  # Convert the image to a tensor (with 3 channels)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet values
    ])
    
    # Apply the transformations and add a batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)  # Move tensor to CPU
    return image_tensor

def predict_mask(image_tensor):
    """ Run the model's prediction and generate the segmentation mask """
    with torch.no_grad():
        output = model(image_tensor)  # Get the model output
        logits = output['out']
        probabilities = torch.sigmoid(logits)
        mask = probabilities.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)  # Convert probabilities to binary mask
    return mask
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file, opacity=0.5):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, %s), rgba(0, 0, 0, %s)), url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % (opacity, opacity, bin_str)
    st.markdown(page_bg_img, unsafe_allow_html=True)
# Set background image with opacity
set_background('Picture1.png', opacity=0.89)


# Set background image with opacity



# Streamlit app interface
st.title("Brain MRI Segmentation")
st.markdown("Upload an image and click **Predict Mask** to see the results.")



# Upload the TIFF image
uploaded_file = st.file_uploader("Choose a TIFF image...", type=["tiff", "tif"])

if uploaded_file is not None:
    try:
        # Load the image using PIL
        image = Image.open(uploaded_file)
        
        # Display a resized version of the image
        st.image(image, caption='Uploaded Image (TIFF)', use_column_width=False, width=300)
        
        # Add a button to trigger prediction
        if st.button("Predict Mask"):
            with st.spinner('Processing...'):
                # Preprocess the image
                image_tensor = preprocess_image(image)
                
                # Predict the segmentation mask
                mask = predict_mask(image_tensor)
                
                # Display the original image and predicted mask side by side
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(image)
                ax[0].set_title('Original Image')
                ax[1].imshow(mask, cmap='gray')
                ax[1].set_title('Predicted Mask')
                ax[0].axis('off')
                ax[1].axis('off')
                
                # Render the plot in Streamlit
                st.pyplot(fig)
            st.success('Prediction complete!')
        
    except Exception as e:
        st.error(f"Error processing image or predicting mask: {e}")
    
    # Additional info
with st.expander("How to use the app"):
    st.write("1. Upload a **TIFF** image.\n"
                 "2. Once uploaded, click the **Predict Mask** button to see the segmented output.\n"
                 "3. Use the expanders for additional tips or details.")

with st.expander("App Information"):
    st.write("This app uses **DeepLabV3** for Brain MRI segmentation, optimized for a 256x256 input size.")