import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.feature import hog
from skimage.filters import gabor, sobel, prewitt, roberts, laplace
from skimage import exposure, measure, feature, filters
from skimage.restoration import denoise_bilateral
from skimage.segmentation import watershed
from skimage.color import label2rgb
from skimage.morphology import erosion, dilation, opening, closing
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, gaussian_filter
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI not available. AI insights will be limited.")

# Configure Gemini if available
if GEMINI_AVAILABLE:
    # You can set your API key in environment variables or Streamlit secrets
    api_key = os.getenv('GOOGLE_API_KEY') or st.secrets.get('GOOGLE_API_KEY', '')
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_CONFIGURED = True
    else:
        GEMINI_CONFIGURED = False
else:
    GEMINI_CONFIGURED = False

# Configure page
st.set_page_config(
    page_title="🚀 Autonomous Landing Site Assessment",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .safety-safe {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .safety-unsafe {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B35;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B35;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Landing Safety Assessment Functions
@st.cache_data
def calculate_surface_slope(image):
    """Calculate surface slope using Sobel operators"""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    slope = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(slope)

@st.cache_data
def calculate_surface_roughness(image):
    """Calculate surface roughness using Laplacian operator"""
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return np.var(laplacian)

@st.cache_data
def calculate_edge_density(image):
    """Calculate edge density for terrain complexity assessment"""
    edges = cv2.Canny(image, 100, 200)
    return np.sum(edges > 0) / edges.size

@st.cache_data
def calculate_texture_uniformity(image):
    """Calculate texture uniformity using Local Binary Pattern variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Simple texture measure using standard deviation in local windows
    kernel = np.ones((9,9)) / 81
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    return np.mean(local_var)

def assess_comprehensive_landing_safety(image):
    """Comprehensive landing safety assessment"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Calculate metrics
    slope = calculate_surface_slope(gray)
    roughness = calculate_surface_roughness(gray)
    edge_density = calculate_edge_density(gray)
    texture_uniformity = calculate_texture_uniformity(image)
    
    # Define thresholds
    max_slope = 15.0
    max_roughness = 1000.0
    max_edge_density = 0.1
    max_texture_variance = 500.0
    
    # Calculate individual scores (0-100)
    slope_score = max(0, 100 - (slope / max_slope) * 100)
    roughness_score = max(0, 100 - (roughness / max_roughness) * 100)
    edge_score = max(0, 100 - (edge_density / max_edge_density) * 100)
    texture_score = max(0, 100 - (texture_uniformity / max_texture_variance) * 100)
    
    # Overall safety score (weighted average)
    overall_score = (slope_score * 0.3 + roughness_score * 0.3 + 
                    edge_score * 0.2 + texture_score * 0.2)
    
    # Safety classification
    if overall_score >= 80:
        safety_status = "EXCELLENT"
        safety_color = "#4CAF50"
    elif overall_score >= 60:
        safety_status = "GOOD"
        safety_color = "#FFC107"
    elif overall_score >= 40:
        safety_status = "MODERATE"
        safety_color = "#FF9800"
    else:
        safety_status = "POOR"
        safety_color = "#f44336"
    
    return {
        'overall_score': overall_score,
        'safety_status': safety_status,
        'safety_color': safety_color,
        'slope': slope,
        'roughness': roughness,
        'edge_density': edge_density,
        'texture_uniformity': texture_uniformity,
        'slope_score': slope_score,
        'roughness_score': roughness_score,
        'edge_score': edge_score,
        'texture_score': texture_score
    }

# Feature Analysis Functions
@st.cache_data
def extract_key_features(image):
    """Extract key visual features for landing assessment"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    features = {}
    
    # SIFT Features
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    features['sift_keypoints'] = len(keypoints)
    features['sift_image'] = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    features['canny_edges'] = edges
    
    # Sobel Gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    features['sobel_magnitude'] = sobel_combined
    
    # Hough Lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
    line_image = image.copy()
    if lines is not None:
        for rho, theta in lines[:min(10, len(lines)), 0]:  # Limit to 10 lines
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
            x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    features['hough_lines'] = line_image
    features['num_lines'] = len(lines) if lines is not None else 0
    
    # Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    features['contours'] = contour_image
    features['num_contours'] = len(contours)
    
    return features

def create_safety_dashboard(safety_results):
    """Create an interactive safety dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Safety Score', 'Individual Metrics', 'Safety Breakdown', 'Risk Assessment'),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Overall Safety Score Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=safety_results['overall_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Safety Score"},
            delta={'reference': 80},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': safety_results['safety_color']},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # Individual Metrics Bar Chart
    metrics = ['Slope', 'Roughness', 'Edge Density', 'Texture']
    scores = [safety_results['slope_score'], safety_results['roughness_score'], 
              safety_results['edge_score'], safety_results['texture_score']]
    
    fig.add_trace(
        go.Bar(x=metrics, y=scores, marker_color=['#FF6B35', '#4ECDC4', '#45B7D1', '#96CEB4']),
        row=1, col=2
    )
    
    # Safety Breakdown Pie Chart
    fig.add_trace(
        go.Pie(labels=metrics, values=scores, hole=0.3),
        row=2, col=1
    )
    
    # Risk Assessment Scatter
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    risk_scores = [safety_results['overall_score'], 100-safety_results['overall_score'], 
                   safety_results['edge_density']*1000, safety_results['roughness']/100]
    
    fig.add_trace(
        go.Scatter(x=risk_levels, y=risk_scores, mode='markers+lines',
                  marker=dict(size=15, color=safety_results['safety_color'])),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Landing Safety Dashboard")
    return fig

# AI Assistant Functions
class AIAssistant:
    def __init__(self):
        self.model = None
        if GEMINI_AVAILABLE and GEMINI_CONFIGURED:
            try:
                self.model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                st.warning(f"Failed to initialize Gemini model: {e}")
                self.model = None
    
    def verify_safety_scores(self, safety_results, features):
        """Use AI to verify and explain safety scores"""
        if not self.model:
            return self._fallback_analysis(safety_results, features)
        
        try:
            prompt = f"""
            As an expert aerospace engineer specializing in autonomous landing systems, please analyze these landing site assessment metrics:

            SAFETY METRICS:
            - Overall Safety Score: {safety_results['overall_score']:.1f}/100
            - Surface Slope: {safety_results['slope']:.2f}° (Score: {safety_results['slope_score']:.1f}/100)
            - Surface Roughness: {safety_results['roughness']:.1f} (Score: {safety_results['roughness_score']:.1f}/100)
            - Edge Density: {safety_results['edge_density']:.3f} (Score: {safety_results['edge_score']:.1f}/100)
            - Texture Uniformity: {safety_results['texture_uniformity']:.1f} (Score: {safety_results['texture_score']:.1f}/100)

            COMPUTER VISION FEATURES:
            - SIFT Keypoints detected: {features['sift_keypoints']}
            - Hough Lines detected: {features['num_lines']}
            - Contours detected: {features['num_contours']}

            Please provide:
            1. VERIFICATION: Are these scores reasonable for landing safety assessment?
            2. EXPLANATION: Detailed explanation of what each metric means for landing safety
            3. RECOMMENDATIONS: Specific recommendations based on these values
            4. RISK ANALYSIS: Potential risks and mitigation strategies
            5. CONFIDENCE: Your confidence level in this assessment (0-100%)

            Keep the response technical but accessible, and focus on practical landing considerations.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(safety_results, features)
    
    def explain_metrics(self, metric_name, value, score):
        """Explain individual metrics in detail"""
        if not self.model:
            return self._fallback_metric_explanation(metric_name, value, score)
        
        try:
            prompt = f"""
            Explain the landing safety metric "{metric_name}" with value {value} and safety score {score:.1f}/100.
            
            Please provide:
            1. What this metric measures
            2. Why it's important for landing safety
            3. How this specific value affects landing risk
            4. What would be ideal values
            5. Potential consequences of this measurement
            
            Keep it concise but informative (2-3 sentences max).
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return self._fallback_metric_explanation(metric_name, value, score)
    
    def chat_response(self, user_question, context):
        """Handle user questions about the analysis"""
        if not self.model:
            return self._fallback_chat_response(user_question, context)
        
        try:
            prompt = f"""
            You are an AI assistant for a Mars landing site assessment system. A user has asked: "{user_question}"
            
            Context from current analysis:
            {context}
            
            Please provide a helpful, accurate response related to:
            - Landing safety assessment
            - Computer vision techniques used
            - Mars terrain analysis
            - Aerospace engineering principles
            
            Keep responses concise and practical.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I'm having trouble processing your question right now. Error: {str(e)}"
    
    def _fallback_analysis(self, safety_results, features):
        """Fallback analysis when AI is not available"""
        score = safety_results['overall_score']
        analysis = f"""
        ## 🔍 Technical Analysis Summary
        
        **Overall Assessment:** {safety_results['safety_status']} (Score: {score:.1f}/100)
        
        ### 📊 Metric Analysis:
        
        **Surface Slope ({safety_results['slope']:.2f}°):**
        - {"✅ Within safe limits" if safety_results['slope'] < 10 else "⚠️ Elevated slope detected"}
        - Slope affects landing stability and rollover risk
        
        **Surface Roughness ({safety_results['roughness']:.1f}):**
        - {"✅ Smooth terrain" if safety_results['roughness'] < 500 else "⚠️ Rough terrain detected"}
        - High roughness can damage landing gear
        
        **Edge Density ({safety_results['edge_density']:.3f}):**
        - {"✅ Simple terrain" if safety_results['edge_density'] < 0.05 else "⚠️ Complex terrain features"}
        - Indicates terrain complexity and potential hazards
        
        **Texture Uniformity ({safety_results['texture_uniformity']:.1f}):**
        - {"✅ Uniform surface" if safety_results['texture_uniformity'] < 300 else "⚠️ Varied surface texture"}
        - Affects surface predictability
        
        ### 🤖 Computer Vision Features:
        - **{features['sift_keypoints']} SIFT keypoints** - Feature richness indicator
        - **{features['num_lines']} linear structures** - Geological features
        - **{features['num_contours']} contours** - Terrain complexity
        
        ### 🎯 Confidence Level: {min(100, max(50, score + 10)):.0f}%
        """
        return analysis
    
    def _fallback_metric_explanation(self, metric_name, value, score):
        """Fallback explanations for metrics"""
        explanations = {
            "Surface Slope": f"Measures terrain incline ({value:.2f}°). Slopes >15° increase rollover risk during landing.",
            "Surface Roughness": f"Quantifies surface irregularities ({value:.1f}). High values indicate potential landing gear damage.",
            "Edge Density": f"Measures terrain complexity ({value:.3f}). High values suggest rocks, craters, or other hazards.",
            "Texture Uniformity": f"Evaluates surface consistency ({value:.1f}). Uniform surfaces are more predictable for landing."
        }
        return explanations.get(metric_name, f"Metric {metric_name}: {value}")
    
    def _fallback_chat_response(self, question, context):
        """Fallback chat responses"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['slope', 'angle', 'incline']):
            return "Surface slope measures the terrain incline. For safe landing, slopes should be <15°. Steeper slopes increase rollover risk."
        elif any(word in question_lower for word in ['rough', 'texture', 'surface']):
            return "Surface roughness indicates terrain irregularities. Smoother surfaces (lower values) are safer for landing gear."
        elif any(word in question_lower for word in ['edge', 'feature', 'complexity']):
            return "Edge density measures terrain complexity. Lower values suggest fewer obstacles like rocks or crater rims."
        elif any(word in question_lower for word in ['sift', 'keypoint', 'feature']):
            return "SIFT keypoints are distinctive image features used for terrain analysis. More keypoints indicate feature-rich terrain."
        elif any(word in question_lower for word in ['safety', 'score', 'assessment']):
            return "The safety score combines multiple metrics: slope (30%), roughness (30%), edge density (20%), and texture (20%)."
        else:
            return "I can help explain landing safety metrics, computer vision features, or terrain analysis. What would you like to know?"

# Initialize AI Assistant
@st.cache_resource
def get_ai_assistant():
    return AIAssistant()

def main():
    # Initialize AI Assistant
    ai_assistant = get_ai_assistant()
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Autonomous Landing Site Assessment</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Computer Vision Analysis for Safe Mars Landing 🔴")
    
    # Add API key input if Gemini is available but not configured
    if GEMINI_AVAILABLE and not GEMINI_CONFIGURED:
        with st.expander("🤖 Configure AI Assistant (Optional)", expanded=False):
            api_key = st.text_input(
                "Enter Google AI API Key for enhanced AI insights:",
                type="password",
                help="Get your free API key from https://makersuite.google.com/app/apikey"
            )
            if api_key:
                genai.configure(api_key=api_key)
                st.success("✅ AI Assistant configured!")
                st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛸 Mission Control")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "📸 Upload Mars Terrain Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a high-resolution image of the landing site"
        )
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("## 🔧 Analysis Options")
        show_features = st.checkbox("🔍 Show Feature Analysis", value=True)
        show_dashboard = st.checkbox("📊 Show Safety Dashboard", value=True)
        show_detailed = st.checkbox("📋 Show Detailed Metrics", value=True)
        show_ai_insights = st.checkbox("🤖 AI-Powered Insights", value=True)
        
        st.markdown("---")
        
        # Safety thresholds
        st.markdown("## ⚙️ Safety Thresholds")
        slope_threshold = st.slider("Max Slope", 5.0, 25.0, 15.0, 0.5)
        roughness_threshold = st.slider("Max Roughness", 100, 2000, 1000, 50)
        
        # AI Assistant Status
        st.markdown("---")
        st.markdown("## 🤖 AI Assistant Status")
        if GEMINI_AVAILABLE and GEMINI_CONFIGURED:
            st.success("✅ Gemini AI Active")
        elif GEMINI_AVAILABLE:
            st.warning("⚠️ Gemini Available (No API Key)")
        else:
            st.info("ℹ️ Basic AI Fallback")
    
    if uploaded_file is not None:
        # Load and display image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract features once for all tabs
        with st.spinner("Extracting features..."):
            safety_results = assess_comprehensive_landing_safety(image)
            features = extract_key_features(image)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Safety Analysis", "🔍 Feature Detection", "📈 Visual Analysis", "🤖 AI Insights", "💬 AI Assistant"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 🖼️ Original Image")
                st.image(image_rgb, use_container_width=True)
            
            with col2:
                st.markdown("### 🛡️ Safety Assessment")
                
                # Perform safety analysis
                with st.spinner("Analyzing landing safety..."):
                    pass  # Already extracted above
                
                # Display safety status
                safety_class = "safety-safe" if safety_results['overall_score'] >= 60 else "safety-unsafe"
                st.markdown(f"""
                <div class="{safety_class}">
                    <h2>🎯 Landing Status: {safety_results['safety_status']}</h2>
                    <h3>Safety Score: {safety_results['overall_score']:.1f}/100</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics with AI explanations
                if show_detailed:
                    st.markdown("#### 📊 Detailed Metrics")
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric(
                            label="🏔️ Surface Slope",
                            value=f"{safety_results['slope']:.2f}°",
                            delta=f"{safety_results['slope_score']:.1f}% safe"
                        )
                        if st.button("🤖 Explain Slope", key="explain_slope"):
                            explanation = ai_assistant.explain_metrics("Surface Slope", safety_results['slope'], safety_results['slope_score'])
                            st.info(explanation)
                        
                        st.metric(
                            label="🌊 Surface Roughness",
                            value=f"{safety_results['roughness']:.1f}",
                            delta=f"{safety_results['roughness_score']:.1f}% safe"
                        )
                        if st.button("🤖 Explain Roughness", key="explain_roughness"):
                            explanation = ai_assistant.explain_metrics("Surface Roughness", safety_results['roughness'], safety_results['roughness_score'])
                            st.info(explanation)
                    
                    with metrics_col2:
                        st.metric(
                            label="🔍 Edge Density",
                            value=f"{safety_results['edge_density']:.3f}",
                            delta=f"{safety_results['edge_score']:.1f}% safe"
                        )
                        if st.button("🤖 Explain Edge Density", key="explain_edge"):
                            explanation = ai_assistant.explain_metrics("Edge Density", safety_results['edge_density'], safety_results['edge_score'])
                            st.info(explanation)
                        
                        st.metric(
                            label="🎨 Texture Variance",
                            value=f"{safety_results['texture_uniformity']:.1f}",
                            delta=f"{safety_results['texture_score']:.1f}% safe"
                        )
                        if st.button("🤖 Explain Texture", key="explain_texture"):
                            explanation = ai_assistant.explain_metrics("Texture Uniformity", safety_results['texture_uniformity'], safety_results['texture_score'])
                            st.info(explanation)
            
            # Safety Dashboard
            if show_dashboard:
                st.markdown("### 📈 Interactive Safety Dashboard")
                dashboard_fig = create_safety_dashboard(safety_results)
                st.plotly_chart(dashboard_fig, use_container_width=True)
                
                # AI Verification of Scores
                if show_ai_insights:
                    st.markdown("### 🤖 AI Score Verification")
                    with st.spinner("AI is analyzing the safety assessment..."):
                        ai_analysis = ai_assistant.verify_safety_scores(safety_results, features)
                        st.markdown(ai_analysis)
        
        with tab2:
            if show_features:
                st.markdown("### 🔍 Computer Vision Feature Analysis")
                
                with st.spinner("Extracting visual features..."):
                    pass  # Already extracted above
                
                # Feature summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 SIFT Keypoints", features['sift_keypoints'])
                with col2:
                    st.metric("📏 Detected Lines", features['num_lines'])
                with col3:
                    st.metric("🔲 Contours", features['num_contours'])
                with col4:
                    st.metric("⚡ Processing", "Complete ✅")
                
                # Feature visualizations
                st.markdown("#### 🎨 Visual Feature Analysis")
                
                feat_col1, feat_col2 = st.columns(2)
                
                with feat_col1:
                    st.markdown("**🎯 SIFT Keypoints**")
                    st.image(features['sift_image'], use_container_width=True)
                    
                    st.markdown("**📏 Hough Lines**")
                    st.image(features['hough_lines'], use_container_width=True)
                
                with feat_col2:
                    st.markdown("**🔍 Edge Detection**")
                    st.image(features['canny_edges'], use_container_width=True)
                    
                    st.markdown("**🔲 Contour Detection**")
                    st.image(features['contours'], use_container_width=True)
        
        with tab3:
            st.markdown("### 📊 Advanced Visual Analysis")
            
            # Gradient analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            grad_col1, grad_col2 = st.columns(2)
            
            with grad_col1:
                st.markdown("**📈 Sobel Gradient Magnitude**")
                sobel_combined = features['sobel_magnitude']
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(sobel_combined, cmap='hot')
                ax.set_title('Surface Gradient Analysis')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                plt.close()
            
            with grad_col2:
                st.markdown("**📊 Intensity Histogram**")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(gray.flatten(), bins=50, alpha=0.7, color='skyblue')
                ax.set_title('Pixel Intensity Distribution')
                ax.set_xlabel('Intensity')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # 3D Surface plot
            st.markdown("**🏔️ 3D Surface Topography**")
            # Downsample for performance
            small_gray = cv2.resize(gray, (100, 100))
            x = np.arange(small_gray.shape[1])
            y = np.arange(small_gray.shape[0])
            X, Y = np.meshgrid(x, y)
            
            fig_3d = go.Figure(data=[go.Surface(z=small_gray, x=X, y=Y, colorscale='viridis')])
            fig_3d.update_layout(
                title='3D Terrain Surface',
                scene=dict(
                    xaxis_title='X Coordinate',
                    yaxis_title='Y Coordinate',
                    zaxis_title='Elevation'
                ),
                height=500
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        with tab4:
            st.markdown("### 🤖 AI-Powered Landing Recommendations")
            
            # Generate AI insights based on analysis
            ai_col1, ai_col2 = st.columns([2, 1])
            
            with ai_col1:
                st.markdown("#### 🧠 Intelligent Analysis")
                
                score = safety_results['overall_score']
                
                if score >= 80:
                    st.success("✅ **EXCELLENT LANDING SITE**")
                    st.write("This location shows optimal characteristics for landing:")
                    st.write("• Low surface slope ensures stable touchdown")
                    st.write("• Minimal roughness reduces landing stress")
                    st.write("• Uniform terrain texture indicates stable surface")
                    recommendation = "PROCEED WITH LANDING"
                elif score >= 60:
                    st.warning("⚠️ **ACCEPTABLE LANDING SITE**")
                    st.write("This location has moderate risk factors:")
                    st.write("• Some terrain irregularities detected")
                    st.write("• Consider alternative sites if available")
                    st.write("• Proceed with enhanced caution")
                    recommendation = "CONDITIONAL LANDING"
                else:
                    st.error("❌ **HIGH RISK LANDING SITE**")
                    st.write("This location presents significant hazards:")
                    st.write("• Excessive surface slope or roughness")
                    st.write("• Complex terrain features detected")
                    st.write("• Alternative site strongly recommended")
                    recommendation = "ABORT LANDING"
                
                st.markdown(f"### 🎯 Final Recommendation: **{recommendation}**")
            
            with ai_col2:
                st.markdown("#### 📊 Risk Factors")
                
                risk_factors = []
                if safety_results['slope'] > 10:
                    risk_factors.append("High slope angle")
                if safety_results['roughness'] > 800:
                    risk_factors.append("Surface roughness")
                if safety_results['edge_density'] > 0.08:
                    risk_factors.append("Complex terrain")
                if features['num_lines'] > 20:
                    risk_factors.append("Linear structures")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"⚠️ {factor}")
                else:
                    st.markdown("✅ No major risk factors detected")
                
                st.markdown("#### 🎯 Confidence Level")
                confidence = min(100, max(50, score + 10))
                st.progress(confidence / 100)
                st.write(f"Analysis Confidence: {confidence:.0f}%")
        
        with tab5:
            st.markdown("### 💬 AI Assistant Chat")
            st.markdown("Ask questions about the landing assessment, metrics, or computer vision techniques!")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat interface
            user_question = st.text_input("Ask the AI Assistant:", placeholder="e.g., Why is the slope score important for landing?")
            
            if st.button("🚀 Ask AI") and user_question:
                # Prepare context for AI
                context = f"""
                Current Analysis Results:
                - Safety Score: {safety_results['overall_score']:.1f}/100
                - Surface Slope: {safety_results['slope']:.2f}°
                - Surface Roughness: {safety_results['roughness']:.1f}
                - Edge Density: {safety_results['edge_density']:.3f}
                - Texture Uniformity: {safety_results['texture_uniformity']:.1f}
                - SIFT Keypoints: {features['sift_keypoints']}
                - Detected Lines: {features['num_lines']}
                - Contours: {features['num_contours']}
                """
                
                # Get AI response
                with st.spinner("AI is thinking..."):
                    ai_response = ai_assistant.chat_response(user_question, context)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("ai", ai_response))
            
            # Display chat history
            for i, (speaker, message) in enumerate(reversed(st.session_state.chat_history[-10:])):  # Show last 10 messages
                if speaker == "user":
                    st.markdown(f"**🧑‍🚀 You:** {message}")
                else:
                    st.markdown(f"**🤖 AI Assistant:** {message}")
                st.markdown("---")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
            
            # Quick question buttons
            st.markdown("#### 🔍 Quick Questions")
            quick_questions = [
                "What does the safety score mean?",
                "How is surface slope calculated?",
                "Why are SIFT keypoints important?",
                "What makes a good landing site?",
                "Explain the edge density metric"
            ]
            
            cols = st.columns(len(quick_questions))
            for i, question in enumerate(quick_questions):
                if cols[i].button(f"❓ {question.split('?')[0]}?", key=f"quick_{i}"):
                    context = f"""
                    Current Analysis Results:
                    - Safety Score: {safety_results['overall_score']:.1f}/100
                    - Surface Slope: {safety_results['slope']:.2f}°
                    - Surface Roughness: {safety_results['roughness']:.1f}
                    - Edge Density: {safety_results['edge_density']:.3f}
                    - Texture Uniformity: {safety_results['texture_uniformity']:.1f}
                    """
                    
                    with st.spinner("AI is thinking..."):
                        ai_response = ai_assistant.chat_response(question, context)
                    
                    st.session_state.chat_history.append(("user", question))
                    st.session_state.chat_history.append(("ai", ai_response))
                    st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>🛸 Welcome to the Autonomous Landing Assessment System</h2>
            <p>Upload a Mars terrain image to begin comprehensive landing safety analysis</p>
            <p>Our AI-powered system will evaluate:</p>
            <ul style="display: inline-block; text-align: left;">
                <li>🏔️ Surface slope and terrain angle</li>
                <li>🌊 Surface roughness and texture</li>
                <li>🔍 Edge density and complexity</li>
                <li>🎯 Feature detection and analysis</li>
                <li>📊 Comprehensive safety scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample images section
        st.markdown("### 📸 Sample Analysis")
        st.write("Here's what our analysis looks like:")
        
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        with demo_col1:
            st.markdown("**🟢 Safe Terrain**")
            st.write("Low slope, uniform texture")
        with demo_col2:
            st.markdown("**🟡 Moderate Risk**")
            st.write("Some irregularities present")
        with demo_col3:
            st.markdown("**🔴 High Risk**")
            st.write("Steep slopes, rough terrain")

if __name__ == "__main__":
    main()
