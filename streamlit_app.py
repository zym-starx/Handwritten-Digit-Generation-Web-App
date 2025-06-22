import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
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

# Your exact model architecture
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = 28 * 28

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.num_classes, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.num_classes, 400),
            nn.ReLU(),
            nn.Linear(400, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, labels):
        """Input: x (batch, 784), labels (batch, 10)"""
        x = torch.cat([x, labels], dim=-1)
        stats = self.encoder(x)
        mu, logvar = stats.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparam Trick"""
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def decode(self, z, labels):
        """Decode latent variable + label"""
        z = torch.cat([z, labels], dim=-1)
        return self.decoder(z)

    def forward(self, x, labels):
        """Complete forward pass"""
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar
    
    def generate(self, digit, num_samples=5):
        """Generate samples for a specific digit"""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Create one-hot encoded labels for the digit
            labels = torch.zeros(num_samples, self.num_classes).to(device)
            labels[:, digit] = 1.0
            
            # Sample from latent space
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Decode to get images
            samples = self.decode(z, labels)
            
            # Reshape to 28x28 images
            return samples.view(-1, 28, 28)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_files = [
        'cvae_model.pth',
        'mnist_digit_generator.pth',
        'model.pth'
    ]
    
    for model_file in model_files:
        try:
            # Initialize model with correct parameters
            model = ConditionalVAE(latent_dim=20, num_classes=10)
            
            # Load the state dict
            state_dict = torch.load(model_file, map_location='cpu')
            model.load_state_dict(state_dict)
            
            st.success(f"✅ Model loaded successfully from {model_file}!")
            return model, True
            
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Error loading {model_file}: {str(e)}")
            continue
    
    # If no model found
    st.info("ℹ️ No trained model found. Using demo mode with pattern generation.")
    return None, False

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
        <p>Generate synthetic MNIST-like images using your trained Conditional VAE</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_model()
    
    # Display mode
    if model_loaded:
        st.markdown('<div class="mode-indicator ml-mode">🧠 ML Mode: Using Your Trained Conditional VAE</div>', 
                   unsafe_allow_html=True)
        mode = "ML"
    else:
        st.markdown('<div class="mode-indicator demo-mode">🎨 Demo Mode: Pattern Generation</div>', 
                   unsafe_allow_html=True)
        mode = "Demo"
    
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
                    # Use your trained model
                    images = model.generate(selected_digit, num_samples=5)
                    images = [img.cpu().numpy() for img in images]
                    generation_method = "Conditional VAE (Your Trained Model)"
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
    
    # Model Information
    st.markdown("---")
    
    with st.expander("📊 Model Architecture"):
        if model_loaded:
            st.write("""
            **Your Conditional VAE Architecture:**
            - **Type:** Conditional Variational Autoencoder
            - **Latent Dimension:** 20
            - **Input Dimension:** 784 (28×28 pixels)
            - **Hidden Layer:** 400 units
            - **Conditioning:** One-hot encoded digit labels (10 classes)
            
            **Encoder:** Input + Label → 400 units → Latent space (μ, σ)
            **Decoder:** Latent + Label → 400 units → 784 pixels
            
            **Training Details:**
            - Framework: PyTorch
            - Dataset: MNIST
            - Loss: Reconstruction + KL Divergence
            - Optimizer: Adam (lr=1e-3)
            """)
        else:
            st.write("""
            **Demo Mode Information:**
            - Using mathematical pattern generation
            - Simulates handwritten digit appearance
            - Upload your trained model for ML-based generation
            """)
    
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
            st.write(f"**Latent dimension:** {model.latent_dim}")
            st.write(f"**Number of classes:** {model.num_classes}")
            try:
                total_params = sum(p.numel() for p in model.parameters())
                st.write(f"**Total parameters:** {total_params:,}")
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
