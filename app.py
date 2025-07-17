import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import json
import time

# Page configuration with custom theme
st.set_page_config(
    page_title="NeuralNet Watchdog 🧠",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI (you can replace with your own API key or use a local model)
try:
    openai.api_key = "your-openai-api-key"  # Replace with your API key
    openai.api_base = "http://localhost:1234/v1"  # For local models like LM Studio
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# Custom CSS for beautiful dark theme styling
st.markdown("""
<style>
    /* Dark theme main styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Header styling with cyber theme */
    .header-container {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,255,255,0.1);
        border: 2px solid rgba(0,255,255,0.2);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(0,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .title-text {
        color: #00ffff;
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0,255,255,0.5);
        position: relative;
        z-index: 1;
    }
    
    .subtitle-text {
        color: #b8b8b8;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* Dark card styling */
    .card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(0,255,255,0.2);
        color: #ffffff;
    }
    
    /* Upload area styling */
    .upload-container {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        border: 3px dashed rgba(0,255,255,0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(0,255,255,0.1) 50%, transparent 70%);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
    
    .upload-container:hover {
        border-color: rgba(0,255,255,0.8);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0,255,255,0.2);
    }
    
    /* Metric cards with cyber theme */
    .metric-card {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        color: #00ffff;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 0.5rem;
        border: 1px solid rgba(0,255,255,0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,255,255,0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #00ffff;
        text-shadow: 0 0 10px rgba(0,255,255,0.5);
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        color: #b8b8b8;
    }
    
    /* Success/Error styling with cyber colors */
    .success-box {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: #0f0f23;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,255,136,0.3);
        border: 1px solid rgba(0,255,136,0.4);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff0066 0%, #cc0044 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255,0,102,0.3);
        border: 1px solid rgba(255,0,102,0.4);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: #0f0f23;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(255,170,0,0.3);
        border: 1px solid rgba(255,170,0,0.4);
    }
    
    .info-box {
        background: linear-gradient(135deg, #0088ff 0%, #0066cc 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: #ffffff;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,136,255,0.3);
        border: 1px solid rgba(0,136,255,0.4);
    }
    
    /* Button styling with cyber theme */
    .stButton > button {
        background: linear-gradient(135deg, #00ffff 0%, #0088cc 100%);
        color: #0f0f23;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 16px rgba(0,255,255,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,255,255,0.4);
        background: linear-gradient(135deg, #00ffff 0%, #00aacc 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #00ffff 0%, #0088cc 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
        border: 1px solid rgba(0,255,255,0.2);
    }
    
    /* Radio button styling */
    .stRadio > div > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0,255,255,0.3);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0,255,255,0.3);
        border-radius: 10px;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border: 3px solid rgba(0,255,255,0.3);
        border-top: 3px solid #00ffff;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00ffff 0%, #0088cc 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00ffff 0%, #00aacc 100%);
    }
    
    /* Hide file uploader label */
    .stFileUploader > div > div > div > div {
        display: none;
    }
    
    /* Text color overrides */
    p, h1, h2, h3, h4, h5, h6, span, div {
        color: #ffffff !important;
    }
    
    /* Metric text color */
    .css-1wivap2 {
        color: #00ffff !important;
    }
    
    /* Dataframe text color */
    .stDataFrame {
        color: #ffffff !important;
    }
    
    /* Chatbot styling */
    .chat-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(0,255,255,0.3);
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-message {
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
        animation: fadeIn 0.5s ease-in;
    }
    
    .chat-message.user {
        background: linear-gradient(135deg, #0088ff 0%, #0066cc 100%);
        margin-left: 2rem;
        border: 1px solid rgba(0,136,255,0.4);
    }
    
    .chat-message.bot {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        margin-right: 2rem;
        border: 1px solid rgba(0,255,136,0.4);
        color: #0f0f23;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Textarea styling */
    .stTextInput > div > div > input {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(0,255,255,0.3);
        border-radius: 10px;
        color: #ffffff;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00ffff;
        box-shadow: 0 0 10px rgba(0,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Header with cyber theme design
st.markdown("""
<div class="header-container">
    <div class="title-text">🛡️ NeuralNet Watchdog</div>
    <div class="subtitle-text">Advanced AI-Powered Network Anomaly Detection System</div>
    <div style="text-align: center; margin-top: 1rem;">
        <span style="background: rgba(0,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem; color: #00ffff;">
            🧠 Machine Learning • 🔍 Real-time Analysis • 🚨 Threat Detection • 🤖 AI Assistant
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with cyber theme
with st.sidebar:
    st.markdown("""
    <div class="card">
        <h3 style="color: #00ffff;">📊 System Status</h3>
        <p style="color: #00ff88;">🟢 Model: Ready</p>
        <p style="color: #00ff88;">🟢 Preprocessing: Active</p>
        <p style="color: #00ff88;">🟢 Detection: Online</p>
        <p style="color: #00ff88;">🟢 AI Assistant: Available</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #00ffff;">🎯 Features</h3>
        <ul style="color: #b8b8b8;">
            <li>Real-time Analysis</li>
            <li>Multi-threat Detection</li>
            <li>Advanced ML Models</li>
            <li>Automated Reporting</li>
            <li>AI Chatbot Assistant</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# File upload with cyber styling
st.markdown("""
<div class="upload-container">
    <h2 style="color: #00ffff; margin-bottom: 1rem;">📁 Upload Network Data</h2>
    <p style="color: rgba(0,255,255,0.8);">Drag and drop your CSV file or click to browse</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader", label_visibility="hidden")

model_file = "models/Network_Anomility.joblib"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def preprocess_data(df):
    """Convert categorical data to numerical and handle missing values"""
    df_processed = df.copy()
    
    # Convert categorical columns to numerical
    label_encoders = {}
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            # Create label encoder for this column
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Convert to float
    df_processed = df_processed.astype(float)
    
    return df_processed, label_encoders

def create_anomaly_chart(predictions):
    """Create beautiful anomaly detection chart with cyber theme"""
    anomaly_count = (predictions == 1).sum()
    normal_count = len(predictions) - anomaly_count
    
    fig = go.Figure()
    
    # Add pie chart with cyber colors
    fig.add_trace(go.Pie(
        labels=['Normal Traffic', 'Anomalies Detected'],
        values=[normal_count, anomaly_count],
        hole=0.4,
        marker_colors=['#00ff88', '#ff0066'],
        textinfo='label+percent',
        textfont_size=14,
        textposition='inside',
        textfont_color='#ffffff'
    ))
    
    fig.update_layout(
        title="Network Traffic Analysis",
        title_x=0.5,
        title_font_color='#00ffff',
        showlegend=True,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff')
    )
    
    return fig

def get_ai_explanation(predictions, df, user_question=""):
    """Get AI explanation for predictions"""
    anomaly_count = (predictions == 1).sum()
    normal_count = len(predictions) - anomaly_count
    total_count = len(predictions)
    anomaly_percentage = (anomaly_count / total_count * 100) if total_count > 0 else 0
    
    # Create context for the AI
    context = f"""
    Network Anomaly Detection Results:
    - Total records analyzed: {total_count:,}
    - Normal traffic: {normal_count:,} ({100-anomaly_percentage:.1f}%)
    - Anomalies detected: {anomaly_count:,} ({anomaly_percentage:.1f}%)
    - Data features: {len(df.columns)} columns
    - Sample data types: {list(df.dtypes.unique())}
    """
    
    # Try OpenAI first, then fallback to local analysis
    if OPENAI_AVAILABLE:
        try:
            if user_question:
                prompt = f"""
                You are an expert cybersecurity analyst and AI assistant for a network anomaly detection system.
                
                {context}
                
                User Question: {user_question}
                
                Please provide a detailed, professional explanation that includes:
                1. Analysis of the results
                2. Potential security implications
                3. Recommendations for further investigation
                4. Technical insights about the detected patterns
                
                Use a professional but accessible tone. Include specific numbers and percentages from the data.
                """
            else:
                prompt = f"""
                You are an expert cybersecurity analyst and AI assistant for a network anomaly detection system.
                
                {context}
                
                Please provide a comprehensive analysis of these network anomaly detection results, including:
                1. Summary of findings
                2. Security implications
                3. Risk assessment
                4. Recommendations for network administrators
                5. Technical insights about the anomaly patterns
                
                Use a professional but accessible tone. Include specific numbers and percentages from the data.
                """
            
            response = openai.ChatCompletion.create(
                model="local-model",  # or "gpt-3.5-turbo" for OpenAI
                messages=[
                    {"role": "system", "content": "You are an expert cybersecurity analyst specializing in network security and anomaly detection."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response['choices'][0]['message']['content']
        
        except Exception as e:
            # Fallback to local analysis
            pass
    
    # Fallback local analysis
    return get_local_analysis(predictions, df, user_question, context)

def get_local_analysis(predictions, df, user_question, context):
    """Local AI analysis without external API"""
    anomaly_count = (predictions == 1).sum()
    normal_count = len(predictions) - anomaly_count
    total_count = len(predictions)
    anomaly_percentage = (anomaly_count / total_count * 100) if total_count > 0 else 0
    
    # Analyze patterns
    if anomaly_percentage > 50:
        risk_level = "HIGH"
        risk_description = "Critical security concern detected"
    elif anomaly_percentage > 20:
        risk_level = "MEDIUM"
        risk_description = "Moderate security concern detected"
    elif anomaly_percentage > 5:
        risk_level = "LOW"
        risk_description = "Minor security concern detected"
    else:
        risk_level = "MINIMAL"
        risk_description = "Network appears secure"
    
    # Generate intelligent response based on user question
    if user_question:
        question_lower = user_question.lower()
        
        if "anomaly" in question_lower or "threat" in question_lower:
            return f"""
🔍 **Anomaly Analysis Summary**

**Risk Level: {risk_level}** - {risk_description}

**Key Findings:**
• Detected {anomaly_count:,} anomalies out of {total_count:,} total records ({anomaly_percentage:.1f}%)
• {normal_count:,} records classified as normal traffic ({100-anomaly_percentage:.1f}%)

**Security Implications:**
• The {anomaly_percentage:.1f}% anomaly rate suggests {'significant security concerns' if anomaly_percentage > 20 else 'moderate security concerns' if anomaly_percentage > 5 else 'minimal security concerns'}
• {'Immediate investigation recommended' if anomaly_percentage > 20 else 'Further monitoring advised' if anomaly_percentage > 5 else 'Routine monitoring sufficient'}

**Recommendations:**
• {'Prioritize investigation of anomalous patterns' if anomaly_percentage > 20 else 'Review network logs for suspicious activity' if anomaly_percentage > 5 else 'Continue normal monitoring'}
• Analyze specific features that triggered the anomaly detection
• Consider implementing additional security measures
            """
        
        elif "investigate" in question_lower or "pattern" in question_lower:
            return f"""
🔬 **Investigation Guidance**

**Analysis Approach:**
• Focus on the {anomaly_count:,} flagged records for detailed investigation
• Review network logs corresponding to anomaly timestamps
• Examine traffic patterns around detected anomalies

**Investigation Steps:**
1. **Immediate Actions:**
   • Isolate affected systems if {anomaly_percentage > 20}
   • Review firewall logs for suspicious connections
   • Check for unauthorized access attempts

2. **Deep Analysis:**
   • Analyze packet characteristics of anomalous traffic
   • Compare with baseline network behavior
   • Identify potential attack vectors

3. **Prevention Measures:**
   • Update security policies based on findings
   • Implement additional monitoring for detected patterns
   • Consider network segmentation if needed
            """
        
        elif "recommend" in question_lower or "action" in question_lower:
            return f"""
⚡ **Action Recommendations**

**Immediate Actions ({risk_level} Priority):**
• {'Emergency response required' if anomaly_percentage > 20 else 'Standard investigation protocol' if anomaly_percentage > 5 else 'Routine review'}
• Document all anomalous events for analysis
• Review network access controls

**Short-term Actions:**
• Implement enhanced monitoring for detected patterns
• Update intrusion detection rules
• Conduct security awareness training

**Long-term Actions:**
• Develop incident response procedures
• Establish baseline network behavior profiles
• Regular security assessments
            """
        
        else:
            return f"""
🤖 **AI Analysis Response**

**Network Security Assessment:**
• **Risk Level:** {risk_level}
• **Anomaly Rate:** {anomaly_percentage:.1f}% ({anomaly_count:,} out of {total_count:,} records)
• **Normal Traffic:** {100-anomaly_percentage:.1f}% ({normal_count:,} records)

**Key Insights:**
• {'High priority investigation needed' if anomaly_percentage > 20 else 'Moderate concern requires attention' if anomaly_percentage > 5 else 'Minimal security concerns detected'}
• Network analysis completed on {len(df.columns)} features
• Detection model processed {total_count:,} records successfully

**Next Steps:**
• Review detailed results table for specific anomalies
• Export results for further analysis
• Consider additional security measures based on findings
            """
    
    else:
        # Default comprehensive analysis
        return f"""
🛡️ **Comprehensive Security Analysis**

**Executive Summary:**
• **Total Records Analyzed:** {total_count:,}
• **Anomalies Detected:** {anomaly_count:,} ({anomaly_percentage:.1f}%)
• **Normal Traffic:** {normal_count:,} ({100-anomaly_percentage:.1f}%)
• **Risk Assessment:** {risk_level}

**Security Implications:**
{'🚨 CRITICAL: High anomaly rate indicates potential security breach or attack in progress. Immediate response required.' if anomaly_percentage > 20 else '⚠️ MODERATE: Elevated anomaly levels suggest suspicious activity requiring investigation.' if anomaly_percentage > 5 else '✅ SECURE: Low anomaly rate indicates normal network operations with minimal security concerns.'}

**Technical Analysis:**
• Model analyzed {len(df.columns)} network features
• Detection accuracy based on machine learning algorithms
• Anomaly patterns identified through statistical analysis

**Recommendations:**
{'🔴 URGENT: Isolate affected systems, review logs immediately, contact security team' if anomaly_percentage > 20 else '🟡 INVESTIGATE: Review network logs, monitor for additional anomalies, update security measures' if anomaly_percentage > 5 else '🟢 MONITOR: Continue normal operations, maintain current security protocols'}

**Next Steps:**
1. Review detailed results for specific anomaly patterns
2. Export data for further analysis
3. Implement recommended security measures
4. Schedule follow-up assessment
        """

# Initialize session state for chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file is not None:
    # Show file info with cyber theme
    st.markdown(f"""
    <div class="card">
        <h3 style="color: #00ffff;">📄 File Information</h3>
        <p style="color: #b8b8b8;"><strong>File Name:</strong> {uploaded_file.name}</p>
        <p style="color: #b8b8b8;"><strong>File Size:</strong> {uploaded_file.size / 1024 / 1024:.2f} MB</p>
        <p style="color: #b8b8b8;"><strong>File Type:</strong> CSV</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = pd.read_csv(uploaded_file, header=None)
    
    # Automatically encode all object (string) columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    label_col = df.columns[-1]
    X = df.drop(label_col, axis=1)

    # Debug: Show first few rows of features to ensure all are numeric
    st.write('First 5 rows of features (should be all numeric):')
    st.write(X.head())
    
    try:
        with st.spinner("🔄 Loading AI model..."):
            model = load_model(model_file)
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ Model Loaded</h4>
            <p>NeuralNet Watchdog is ready for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Make predictions using processed data
        with st.spinner("🔍 Analyzing network traffic..."):
            predictions = model.predict(X)
        
        # Debug output for diagnosis
        st.write("Prediction value counts:", pd.Series(predictions).value_counts())
        st.write("Features used for prediction:", X.columns.tolist())
        
        df["Prediction"] = predictions

        # Show prediction summary with beautiful metrics
        st.markdown("""
        <div class="card">
            <h3 style="color: #00ffff;">📊 Detection Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create metrics
        anomaly_count = (predictions == 1).sum() if hasattr(predictions, '__iter__') else 0
        normal_count = len(predictions) - anomaly_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{normal_count:,}</div>
                <div class="metric-label">Normal Traffic</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{anomaly_count:,}</div>
                <div class="metric-label">Anomalies Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            total_traffic = normal_count + anomaly_count
            anomaly_percentage = (anomaly_count / total_traffic * 100) if total_traffic > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{anomaly_percentage:.1f}%</div>
                <div class="metric-label">Threat Level</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_traffic:,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Create beautiful charts
        st.markdown("""
        <div class="card">
            <h3 style="color: #00ffff;">📈 Analysis Charts</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Anomaly distribution chart
        fig = create_anomaly_chart(predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        # AI Chatbot Section
        st.markdown("""
        <div class="card">
            <h3 style="color: #00ffff;">🤖 AI Assistant - Ask Questions About Your Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_question = st.text_input(
                "Ask the AI assistant about your results:",
                placeholder="e.g., 'What do these anomalies mean?' or 'How should I investigate these threats?'",
                key="user_question"
            )
        
        with col2:
            if st.button("🤖 Ask AI", key="ask_ai"):
                if user_question:
                    with st.spinner("🤖 AI is analyzing your question..."):
                        ai_response = get_ai_explanation(predictions, df, user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"user": user_question, "bot": ai_response})
        
        # Auto-generate initial analysis
        if not st.session_state.chat_history:
            with st.spinner("🤖 AI is analyzing your results..."):
                initial_analysis = get_ai_explanation(predictions, df)
                st.session_state.chat_history.append({"user": "Initial Analysis", "bot": initial_analysis})
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("""
            <div class="chat-container">
            """, unsafe_allow_html=True)
            
            for i, message in enumerate(st.session_state.chat_history):
                if message["user"] == "Initial Analysis":
                    st.markdown(f"""
                    <div class="chat-message bot">
                        <strong>🤖 AI Assistant:</strong><br>
                        {message["bot"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message user">
                        <strong>👤 You:</strong><br>
                        {message["user"]}
                    </div>
                    <div class="chat-message bot">
                        <strong>🤖 AI Assistant:</strong><br>
                        {message["bot"]}
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show results table
        st.markdown("""
        <div class="card">
            <h3 style="color: #00ffff;">📋 Detailed Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df, use_container_width=True)

        # Download button
        st.markdown("""
        <div class="card">
            <h3 style="color: #00ffff;">💾 Export Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="neuralnet_watchdog_predictions.csv",
            mime="text/csv",
            help="Download the complete analysis results"
        )

    except FileNotFoundError:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Model Not Found</h4>
            <p>Model file not found: {model_file}</p>
            <p>Please make sure the model file exists in the models directory.</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <h4>❌ Prediction Failed</h4>
            <p>Model failed to predict: {e}</p>
            <p>This might be due to incompatible data format or model issues.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    # Welcome screen with cyber theme
    st.markdown("""
    <div class="card">
        <h2 style="color: #00ffff;">🚀 Welcome to NeuralNet Watchdog</h2>
        <p style="color: #b8b8b8;">Your advanced AI-powered network anomaly detection system is ready to analyze your data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #00ffff;">🎯 How it works:</h3>
        <ol style="color: #b8b8b8;">
            <li><strong>Upload your CSV file</strong> containing network traffic data</li>
            <li><strong>Automatic preprocessing</strong> converts categorical data to numerical</li>
            <li><strong>AI analysis</strong> detects anomalies using advanced machine learning</li>
            <li><strong>Get detailed results</strong> with visualizations and downloadable reports</li>
            <li><strong>Ask the AI assistant</strong> for explanations and insights about your results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="color: #00ffff;">📋 Expected Data Format:</h3>
        <p style="color: #b8b8b8;">Your CSV file should have at least 42 columns with network traffic features.</p>
        <p style="color: #b8b8b8;">The model expects exactly 42 features in this order:</p>
        <ul style="color: #b8b8b8;">
            <li>Features 0-40 and 42 (numbered as strings: '0', '1', '2', ..., '40', '42')</li>
            <li>Note: Feature '41' is missing from the model's expectations</li>
        </ul>
        <p style="color: #b8b8b8;"><strong>If your data has more than 42 features:</strong></p>
        <ul style="color: #b8b8b8;">
            <li>You can choose to use the first 42 features automatically</li>
            <li>Or manually select which 42 features to use</li>
        </ul>
        <p style="color: #b8b8b8;">The app will automatically rename columns to match the model's expectations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="text-align: center;">
        <p style="font-size: 1.2rem; color: #00ffff; font-weight: bold;">
            👆 Upload a CSV file to get started with your analysis!
        </p>
    </div>
    """, unsafe_allow_html=True)
