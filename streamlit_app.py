import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Configure page
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .digit-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .generated-image {
        text-align: center;
        padding: 0.5rem;
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Model Definition (same as training script)
class SimpleConditionalVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, latent_dim=16, num_classes=10):
        super(SimpleConditionalVAE, self).__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def decode(self, z, y):
        y_onehot = F.one_hot(y, self.num_classes).float()
        input_with_label = torch.cat([z, y_onehot], dim=1)
        return self.decoder(input_with_label)
    
    def generate(self, digit, num_samples=5):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            y = torch.tensor([digit] * num_samples).to(device)
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z, y)
            return samples.view(-1, 28, 28)

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    try:
        model = SimpleConditionalVAE()
        
        # Try to load the model file
        try:
            checkpoint = torch.load('mnist_digit_generator.pth', map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, True
        except FileNotFoundError:
            # If model file doesn't exist, return untrained model for demo
            st.warning("⚠️ Model file not found. Using demo mode with random generation.")
            return model, False
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

def generate_demo_digit(digit):
    """Generate demo digit images when model isn't available"""
    images = []
    for i in range(5):
        # Create a simple synthetic digit image for demo
        img = np.zeros((28, 28))
        
        # Add some noise and basic shape
        noise = np.random.normal(0, 0.1, (28, 28))
        
        # Create a simple digit-like pattern
        if digit == 0:
            img[8:20, 10:18] = 0.8
            img[10:18, 12:16] = 0.2
        elif digit == 1:
            img[6:22, 13:15] = 0.8
        elif digit == 2:
            img[8:12, 8:20] = 0.8
            img[12:16, 12:20] = 0.8
            img[16:20, 8:20] = 0.8
        else:
            # Generic pattern for other digits
            center_x, center_y = 14, 14
            for x in range(28):
                for y in range(28):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if 6 < dist < 10:
                        img[x, y] = 0.7
        
        img = np.clip(img + noise, 0, 1)
        images.append(img)
    
    return images

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string for display"""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✍️ Handwritten Digit Image Generator</h1>
        <p>Generate synthetic MNIST-like images using your trained model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_model()
    
    if model is None:
        st.error("Failed to initialize model. Please check your setup.")
        return
    
    # Model status
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.info("ℹ️ Running in demo mode. Upload your trained model for actual generation.")
    
    # User input
    st.subheader("🎯 Generate Digits")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        selected_digit = st.selectbox(
            "Choose a digit to generate (0-9):",
            options=list(range(10)),
            index=2,  # Default to digit 2
            help="Select which digit you want to generate"
        )
        
        generate_button = st.button("🎨 Generate Images", use_container_width=True)
    
    # Generation
    if generate_button or 'generated_images' not in st.session_state:
        with st.spinner("🔄 Generating images..."):
            if model_loaded:
                try:
                    # Generate using the trained model
                    images = model.generate(selected_digit, num_samples=5)
                    images = [img.numpy() for img in images]
                except Exception as e:
                    st.error(f"Error generating images: {e}")
                    images = generate_demo_digit(selected_digit)
            else:
                # Use demo generation
                images = generate_demo_digit(selected_digit)
            
            st.session_state.generated_images = images
            st.session_state.current_digit = selected_digit
    
    # Display results
    if 'generated_images' in st.session_state:
        st.subheader(f"📸 Generated Images of Digit {st.session_state.current_digit}")
        
        # Display images in a row
        cols = st.columns(5)
        
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i]:
                st.markdown(f"**Sample {i+1}**")
                
                # Convert to base64 for better display
                img_b64 = numpy_to_base64(img)
                st.markdown(
                    f'<img src="{img_b64}" style="width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">',
                    unsafe_allow_html=True
                )
    
    # Additional features
    st.markdown("---")
    
    with st.expander("📊 Model Information"):
        st.write("""
        **Model Architecture:** Conditional Variational Autoencoder (CVAE)
        - **Input Dimension:** 784 (28×28 pixels)
        - **Hidden Dimension:** 256
        - **Latent Dimension:** 16
        - **Output:** 28×28 grayscale images
        
        **Training Details:**
        - **Dataset:** MNIST (60,000 training samples)
        - **Framework:** PyTorch
        - **Training Environment:** Google Colab with T4 GPU
        """)
    
    with st.expander("🚀 Deployment Information"):
        st.write("""
        **This app is hosted for free and accessible for 2+ weeks through:**
        - Streamlit Community Cloud (free hosting)
        - GitHub integration for continuous deployment
        - No server maintenance required
        
        **Features:**
        - Real-time digit generation
        - Interactive web interface
        - Mobile-friendly responsive design
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🔬 Built with PyTorch & Streamlit | 🎓 MNIST Digit Generation Project"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
