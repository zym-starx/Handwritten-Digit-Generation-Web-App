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
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    .mode-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .ml-mode {
        background: #28a745;
        color: white;
    }
    
    .demo-mode {
        background: #ffc107;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Model Definition - Updated to handle different architectures
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

# Alternative model architecture (in case the saved model is different)
class AlternativeVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_classes=10):
        super(AlternativeVAE, self).__init__()
        
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
    """Load the trained model with multiple fallback strategies"""
    model_files = [
        'mnist_digit_generator.pth',
        'final_digit_generator_model.pth',
        'model.pth',
        'checkpoint.pth'
    ]
    
    for model_file in model_files:
        try:
            # Try to load the checkpoint
            checkpoint = torch.load(model_file, map_location='cpu')
            
            # Strategy 1: Checkpoint format with model_state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Try different model architectures
                for ModelClass, params in [
                    (SimpleConditionalVAE, {'hidden_dim': 256, 'latent_dim': 16}),
                    (SimpleConditionalVAE, {'hidden_dim': 400, 'latent_dim': 20}),
                    (AlternativeVAE, {'hidden_dim': 400, 'latent_dim': 20}),
                    (AlternativeVAE, {'hidden_dim': 256, 'latent_dim': 16})
                ]:
                    try:
                        model = ModelClass(**params)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        st.success(f"✅ Model loaded from {model_file} (checkpoint format)")
                        return model, True, "ML"
                    except:
                        continue
            
            # Strategy 2: Direct state dict
            elif isinstance(checkpoint, dict):
                for ModelClass, params in [
                    (SimpleConditionalVAE, {'hidden_dim': 256, 'latent_dim': 16}),
                    (SimpleConditionalVAE, {'hidden_dim': 400, 'latent_dim': 20}),
                    (AlternativeVAE, {'hidden_dim': 400, 'latent_dim': 20}),
                    (AlternativeVAE, {'hidden_dim': 256, 'latent_dim': 16})
                ]:
                    try:
                        model = ModelClass(**params)
                        model.load_state_dict(checkpoint)
                        st.success(f"✅ Model loaded from {model_file} (state dict format)")
                        return model, True, "ML"
                    except:
                        continue
            
            # Strategy 3: Entire model object
            else:
                try:
                    model = checkpoint
                    if hasattr(model, 'generate'):
                        st.success(f"✅ Model loaded from {model_file} (full model format)")
                        return model, True, "ML"
                except:
                    continue
                    
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Error loading {model_file}: {str(e)}")
            continue
    
    # If all model loading fails, show info and use demo mode
    st.info("ℹ️ No compatible model found. Using demo mode with pattern generation.")
    return None, False, "Demo"

def generate_demo_digit(digit, variation=0):
    """Generate demo digit when model isn't available"""
    img = np.zeros((28, 28))
    np.random.seed(42 + digit * 10 + variation)
    
    if digit == 0:
        center_x, center_y = 14, 14
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if 6 < dist < 10:
                    img[x, y] = 0.8 + np.random.normal(0, 0.1)
    elif digit == 1:
        start_col = 12 + np.random.randint(-2, 3)
        width = 2 + np.random.randint(0, 2)
        img[4:24, start_col:start_col+width] = 0.8
    elif digit == 2:
        img[4:8, 6:22] = 0.8
        img[8:14, 14:22] = 0.8
        img[14:18, 6:14] = 0.8
        img[18:22, 6:22] = 0.8
    elif digit == 3:
        img[4:8, 6:20] = 0.8
        img[12:16, 8:18] = 0.8
        img[20:24, 6:20] = 0.8
        img[4:24, 18:22] = 0.8
    elif digit == 4:
        img[4:20, 6:10] = 0.8
        img[12:16, 6:22] = 0.8
        img[4:24, 18:22] = 0.8
    elif digit == 5:
        img[4:8, 6:22] = 0.8
        img[4:14, 6:10] = 0.8
        img[12:16, 6:18] = 0.8
        img[14:24, 18:22] = 0.8
        img[20:24, 6:22] = 0.8
    elif digit == 6:
        center_x, center_y = 14, 14
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if 6 < dist < 10 and x > 8:
                    img[x, y] = 0.8
        img[8:24, 6:10] = 0.8
    elif digit == 7:
        img[4:8, 6:22] = 0.8
        for i in range(20):
            x = 8 + i
            y = 20 - i // 2
            if 0 <= x < 28 and 0 <= y < 28:
                img[x, y:y+2] = 0.8
    elif digit == 8:
        for x in range(28):
            for y in range(28):
                dist1 = np.sqrt((x - 10)**2 + (y - 14)**2)
                dist2 = np.sqrt((x - 18)**2 + (y - 14)**2)
                if (4 < dist1 < 6) or (4 < dist2 < 6):
                    img[x, y] = 0.8
    elif digit == 9:
        center_x, center_y = 10, 14
        for x in range(28):
            for y in range(28):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if 4 < dist < 7:
                    img[x, y] = 0.8
        img[10:24, 18:22] = 0.8
    
    noise = np.random.normal(0, 0.05, (28, 28))
    img = np.clip(img + noise, 0, 1)
    return img

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
    model, model_loaded, mode = load_model()
    
    # Display mode
    if mode == "ML":
        st.markdown('<div class="mode-indicator ml-mode">🧠 ML Mode: Using Trained Neural Network</div>', 
                   unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-indicator demo-mode">🎨 Demo Mode: Pattern Generation</div>', 
                   unsafe_allow_html=True)
    
    # User input
    st.subheader("🎯 Generate Digits")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        selected_digit = st.selectbox(
            "Choose a digit to generate (0-9):",
            options=list(range(10)),
            index=2,
            help="Select which digit you want to generate"
        )
        
        generate_button = st.button("🎨 Generate Images", use_container_width=True)
    
    # Generation
    if generate_button or 'generated_images' not in st.session_state:
        with st.spinner("🔄 Generating images..."):
            if model_loaded and model:
                try:
                    # Use ML model
                    images = model.generate(selected_digit, num_samples=5)
                    images = [img.numpy() for img in images]
                    generation_method = "Neural Network"
                except Exception as e:
                    st.warning(f"ML generation failed: {e}. Using demo mode.")
                    images = [generate_demo_digit(selected_digit, i) for i in range(5)]
                    generation_method = "Pattern Generation (Fallback)"
            else:
                # Use demo generation
                images = [generate_demo_digit(selected_digit, i) for i in range(5)]
                generation_method = "Pattern Generation"
            
            st.session_state.generated_images = images
            st.session_state.current_digit = selected_digit
            st.session_state.generation_method = generation_method
    
    # Display results
    if 'generated_images' in st.session_state:
        st.subheader(f"📸 Generated Images of Digit {st.session_state.current_digit}")
        st.caption(f"Method: {st.session_state.generation_method}")
        
        cols = st.columns(5)
        
        for i, img in enumerate(st.session_state.generated_images):
            with cols[i]:
                st.markdown(f"**Sample {i+1}**")
                img_b64 = numpy_to_base64(img)
                st.markdown(
                    f'<img src="{img_b64}" style="width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">',
                    unsafe_allow_html=True
                )
    
    # Debug info
    with st.expander("🔧 Debug Information"):
        st.write("**Available files in repository:**")
        import os
        files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if files:
            st.write(f"Found model files: {files}")
        else:
            st.write("No .pth files found in repository")
            
        st.write(f"**Current mode:** {mode}")
        st.write(f"**Model loaded:** {model_loaded}")
        
        if model:
            st.write(f"**Model type:** {type(model).__name__}")
            try:
                st.write(f"**Model parameters:** {sum(p.numel() for p in model.parameters()):,}")
            except:
                st.write("Could not count parameters")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        f"🔬 Built with PyTorch & Streamlit | 🎓 MNIST Project | 🤖 {mode} Mode"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
