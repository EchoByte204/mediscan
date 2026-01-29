"""
MediScan AI - COVID-19 Diagnosis Web Application
Professional Streamlit interface with Grad-CAM explainability
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import io

from src.models.model import create_model
from src.data.preprocessing import DataAugmentation
from src.explainability.gradcam import visualize_gradcam

import urllib.request
import os

def download_model_if_needed():
    """Download model from GitHub release if not present"""
    model_path = 'models/saved/resnet50_best.pth'
    
    # Create directory if doesn't exist
    os.makedirs('models/saved', exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.info("📥 Downloading model (first time only, ~350 MB)...")
        st.info("This may take 2-3 minutes. Please wait...")
        
        try:
            # Download from GitHub release
            url = "https://github.com/EchoByte204/mediscan/releases/download/v1.0.0/resnet50_best.pth"
            
            # Download with progress
            urllib.request.urlretrieve(url, model_path)
            st.success("✅ Model downloaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.info("Please download manually from GitHub releases and place in models/saved/")
            return None
    
    return model_path


def validate_xray_image(image):
    """
    Simple two-stage validation
    Stage 1: Reject obvious non-medical images
    Stage 2: Warn about quality issues
    """
    import numpy as np
    from PIL import ImageStat
    
    gray_image = image.convert('L')
    stat = ImageStat.Stat(gray_image)
    
    width, height = image.size
    mean_brightness = stat.mean[0]
    std_brightness = stat.stddev[0]
    
    # Check RGB variance (main color detector)
    if image.mode == 'RGB':
        r, g, b = image.split()
        r_arr = np.array(r)
        g_arr = np.array(g)
        b_arr = np.array(b)
        
        # Simple check: are R, G, B very different?
        color_diff = np.mean(np.abs(r_arr - g_arr)) + \
                     np.mean(np.abs(g_arr - b_arr)) + \
                     np.mean(np.abs(r_arr - b_arr))
        color_diff /= 3
    else:
        color_diff = 0
    
    # STAGE 1: REJECT OBVIOUS NON-MEDICAL IMAGES
    critical_issues = []
    
    # Strong color = not X-ray
    if color_diff > 20:
        critical_issues.append("❌ Image contains strong colors (not a grayscale X-ray)")
    
    # Pure white/black (screenshots/documents)
    img_array = np.array(gray_image)
    white_ratio = np.sum(img_array > 250) / img_array.size
    black_ratio = np.sum(img_array < 5) / img_array.size
    
    if white_ratio > 0.5 or black_ratio > 0.5:
        critical_issues.append("❌ Image is mostly white/black (document or screenshot)")
    
    # Too small
    if width < 100 or height < 100:
        critical_issues.append("❌ Image resolution too low")
    
    # STAGE 2: QUALITY WARNINGS (don't block)
    warnings = []
    
    if std_brightness < 20:
        warnings.append("⚠️ Low contrast - results may be inaccurate")
    
    if mean_brightness < 30 or mean_brightness > 230:
        warnings.append("⚠️ Unusual brightness")
    
    # DECISION
    is_valid = len(critical_issues) == 0
    
    if not is_valid:
        message = "❌ **This is NOT a medical X-ray image.**"
        all_issues = critical_issues
    else:
        if warnings:
            message = "✅ Image accepted (with quality warnings)"
            all_issues = warnings
        else:
            message = "✅ Image validated - appears to be a medical X-ray"
            all_issues = []
    
    return is_valid, message, all_issues


# Page configuration
st.set_page_config(
    page_title="MediScan AI - COVID-19 Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.device = None

@st.cache_resource
def load_model():
    """Load trained model (cached)"""
    
    # Download model if needed
    MODEL_PATH = download_model_if_needed()
    
    if MODEL_PATH is None:
        return None, None, None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    MODEL_NAME = 'resnet50'
    NUM_CLASSES = 4
    
    # Create and load model
    model = create_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device, checkpoint['best_val_acc']

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/coronavirus.png", width=80)
    st.title("🏥 MediScan AI")
    st.markdown("---")
    
    st.markdown("### About")
    st.info(
        "AI-powered COVID-19 diagnosis system using deep learning "
        "with explainable Grad-CAM visualizations."
    )
    
    # Load model button
    if st.button("🔄 Load AI Model", type="primary", use_container_width=True):
        with st.spinner("Loading model..."):
            model, device, val_acc = load_model()
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.val_acc = val_acc
            st.session_state.model_loaded = True
        st.success(f"✅ Model loaded! Validation Accuracy: {val_acc:.2f}%")
    
    if st.session_state.model_loaded:
        st.markdown("---")
        st.markdown("### Model Information")
        st.metric("Architecture", "ResNet50")
        st.metric("Validation Accuracy", f"{st.session_state.val_acc:.2f}%")
        st.metric("Test Accuracy", "95.56%")
        st.metric("ROC-AUC", "0.9944")
        
        st.markdown("---")
        st.markdown("### Performance")
        st.markdown("**COVID-19 Detection:**")
        st.markdown("- Sensitivity: 97.61%")
        st.markdown("- Specificity: 99.73%")
        st.markdown("- Precision: 98.70%")
    
    st.markdown("---")
    st.caption("⚕️ **Medical Disclaimer:** This is an educational tool. "
               "Always consult healthcare professionals for medical decisions.")

# Main content
st.markdown('<h1 class="main-header">🏥 MediScan AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered COVID-19 Diagnosis with Explainable AI</p>', 
            unsafe_allow_html=True)

if not st.session_state.model_loaded:
    st.warning("👈 Please load the AI model from the sidebar to begin analysis")
    
    # Show example images
    st.markdown("---")
    st.markdown("### 📸 How It Works")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1️⃣ Upload X-Ray")
        st.info("Upload a chest X-ray image in PNG, JPG, or JPEG format")
    with col2:
        st.markdown("#### 2️⃣ AI Analysis")
        st.info("Deep learning model analyzes the image for COVID-19 patterns")
    with col3:
        st.markdown("#### 3️⃣ Visual Explanation")
        st.info("Grad-CAM highlights regions that influenced the diagnosis")
    
else:
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a chest X-ray image for COVID-19 diagnosis"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original image and validation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original X-Ray")
            st.image(image, use_column_width=True)
            
            # VALIDATE IMAGE
            with st.spinner("🔍 Validating image..."):
                is_valid, validation_message, issues = validate_xray_image(image)
            
            # Show validation result
            if is_valid:
                st.success(validation_message)
            else:
                st.error(validation_message)
                if issues:
                    st.markdown("**Issues detected:**")
                    for issue in issues:
                        st.markdown(f"- {issue}")
        
        # Show analyze button only if valid
        if is_valid:
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing the image..."):
                    # Get transform
                    transform = DataAugmentation.get_val_transform()
                    
                    # Save temporary file
                    temp_path = Path("temp_image.png")
                    image.save(temp_path)
                    
                    # Perform analysis
                    CLASS_NAMES = ['COVID-19', 'Lung Opacity', 'Normal', 'Viral Pneumonia']
                    
                    results = visualize_gradcam(
                        model=st.session_state.model,
                        image_path=str(temp_path),
                        transform=transform,
                        device=st.session_state.device,
                        class_names=CLASS_NAMES,
                        model_name='resnet50',
                        save_path=None
                    )
                    
                    # Clean up
                    temp_path.unlink()
                
                with col2:
                    st.markdown("### 🎯 AI Explanation (Grad-CAM)")
                    st.image(results['overlay'], use_column_width=True)
                    st.caption("🔴 Red regions indicate areas that influenced the diagnosis")
                
                # Results section
                st.markdown("---")
                st.markdown("## 📊 Diagnosis Results")
                
                # Main prediction
                prediction = results['predicted_label']
                confidence = results['confidence']
                
                # LOW CONFIDENCE WARNING
                if confidence < 70:
                    st.warning(f"""
                    ⚠️ **LOW CONFIDENCE WARNING**
                    
                    The AI's confidence is only **{confidence:.1f}%**, which is relatively low.
                    
                    **Possible reasons:**
                    - Image quality may be poor or unclear
                    - The X-ray might be non-standard or unusual
                    - This case requires expert medical review
                    
                    **⚠️ Do NOT rely solely on this result. Consult a qualified healthcare professional immediately.**
                    """)
                
                # Color coding
                if prediction == 'COVID-19':
                    alert_type = "error"
                    emoji = "🦠"
                    color = "#FF6B6B"
                elif prediction == 'Viral Pneumonia':
                    alert_type = "warning"
                    emoji = "⚠️"
                    color = "#FFA07A"
                elif prediction == 'Lung Opacity':
                    alert_type = "warning"
                    emoji = "⚠️"
                    color = "#FFD93D"
                else:
                    alert_type = "success"
                    emoji = "✅"
                    color = "#6BCF7F"
                
                # Display prediction
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric("Prediction", f"{emoji} {prediction}")
                with col4:
                    st.metric("Confidence", f"{confidence:.2f}%")
                with col5:
                    risk = "High" if prediction in ['COVID-19', 'Viral Pneumonia'] else \
                           "Moderate" if prediction == 'Lung Opacity' else "Low"
                    st.metric("Risk Level", risk)
                
                # Detailed analysis
                st.markdown("---")
                st.markdown("### 📈 Confidence Distribution")
                
                # Get all probabilities
                input_tensor = transform(image).unsqueeze(0).to(st.session_state.device)
                with torch.no_grad():
                    output = st.session_state.model(input_tensor)
                    probs = torch.softmax(output, dim=1)[0].cpu().numpy() * 100
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(CLASS_NAMES, probs, color=['#FF6B6B', '#FFD93D', '#6BCF7F', '#FFA07A'])
                ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
                ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 100)
                
                # Add value labels
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    ax.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')
                
                ax.grid(axis='x', alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Clinical interpretation
                st.markdown("---")
                st.markdown("### 🩺 Clinical Interpretation")
                
                if prediction == 'COVID-19':
                    st.error(
                        f"⚠️ **COVID-19 Detected** ({confidence:.1f}% confidence)\n\n"
                        "**Findings:** The AI has identified patterns consistent with COVID-19 infection. "
                        "The Grad-CAM visualization shows the specific lung regions that influenced this diagnosis.\n\n"
                        "**Recommendation:** \n"
                        "- Immediate isolation recommended\n"
                        "- RT-PCR test confirmation required\n"
                        "- Consult with healthcare provider immediately\n"
                        "- Follow local health authority guidelines"
                    )
                elif prediction == 'Viral Pneumonia':
                    st.warning(
                        f"⚠️ **Viral Pneumonia Detected** ({confidence:.1f}% confidence)\n\n"
                        "**Findings:** The AI has detected patterns consistent with viral pneumonia.\n\n"
                        "**Recommendation:**\n"
                        "- Medical evaluation required\n"
                        "- Further diagnostic tests recommended\n"
                        "- Consult pulmonologist for treatment plan"
                    )
                elif prediction == 'Lung Opacity':
                    st.warning(
                        f"⚠️ **Lung Opacity Detected** ({confidence:.1f}% confidence)\n\n"
                        "**Findings:** The AI has identified lung opacities that may indicate inflammation or infection.\n\n"
                        "**Recommendation:**\n"
                        "- Follow-up with healthcare provider\n"
                        "- Additional imaging or tests may be needed\n"
                        "- Monitor symptoms"
                    )
                else:
                    st.success(
                        f"✅ **No Significant Abnormalities Detected** ({confidence:.1f}% confidence)\n\n"
                        "**Findings:** The chest X-ray appears normal based on AI analysis.\n\n"
                        "**Note:** This does not rule out all conditions. Continue regular health monitoring "
                        "and consult healthcare professionals if symptoms develop."
                    )
                
                # Additional visualizations
                st.markdown("---")
                st.markdown("### 🔬 Additional Visualizations")
                
                col6, col7, col8 = st.columns(3)
                
                with col6:
                    st.markdown("#### Original Image")
                    st.image(results['original_image'], use_column_width=True)
                
                with col7:
                    st.markdown("#### Heatmap Only")
                    st.image(results['heatmap'], use_column_width=True)
                
                with col8:
                    st.markdown("#### Overlay")
                    st.image(results['overlay'], use_column_width=True)
                
                # Download section
                st.markdown("---")
                col9, col10 = st.columns(2)
                
                with col9:
                    # Create downloadable report
                    report = f"""
MEDISCAN AI - DIAGNOSTIC REPORT
================================

Prediction: {prediction}
Confidence: {confidence:.2f}%
Risk Level: {risk}

Class Probabilities:
{chr(10).join([f'  {name}: {prob:.2f}%' for name, prob in zip(CLASS_NAMES, probs)])}

Model Performance:
  Validation Accuracy: {st.session_state.val_acc:.2f}%
  Test Accuracy: 95.56%
  ROC-AUC: 0.9944

DISCLAIMER: This is an AI-assisted analysis tool for educational 
purposes. It should not replace professional medical diagnosis. 
Always consult qualified healthcare providers for medical decisions.

Generated by MediScan AI
"""
                    st.download_button(
                        label="📄 Download Report",
                        data=report,
                        file_name="mediscan_report.txt",
                        mime="text/plain"
                    )
                
                with col10:
                    # Save overlay image
                    buf = io.BytesIO()
                    Image.fromarray(results['overlay']).save(buf, format='PNG')
                    st.download_button(
                        label="🖼️ Download Visualization",
                        data=buf.getvalue(),
                        file_name="gradcam_overlay.png",
                        mime="image/png"
                    )
        
        else:
            # IMAGE NOT VALID - Show error and instructions
            st.error("❌ **Cannot analyze this image**")
            
            st.warning("""
            **📋 Please upload a valid chest X-ray image.**
            
            ✅ **Accepted:**
            - Grayscale medical X-ray images
            - Front-facing chest views (PA or AP)
            - PNG, JPG, or JPEG format
            - Minimum resolution: 224×224 pixels
            
            ❌ **NOT Accepted:**
            - Color photos, graphics, or screenshots
            - Diagrams or illustrations
            - CT scans or MRI images
            - Non-chest X-rays
            """)
            
            # Show example
            with st.expander("📸 See example of valid chest X-ray"):
                st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/Chest_Xray_PA_3-8-2010.png/300px-Chest_Xray_PA_3-8-2010.png")
                st.caption("Example of a valid chest X-ray (PA view)")
            
            # Disable analyze button
            st.button("🔍 Analyze Image", type="primary", disabled=True, use_container_width=True)
            st.caption("⬆️ Please upload a valid chest X-ray image to enable analysis")
    
    else:
        st.info("👆 Please upload a chest X-ray image to begin analysis")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Made with ❤️ for medical AI research | "
    "ResNet50 Model | 95.56% Test Accuracy | 0.9944 ROC-AUC"
    "</div>",
    unsafe_allow_html=True
)