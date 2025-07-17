import streamlit as st

st.set_page_config(
    page_title="HTML Test",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 HTML Rendering Test")

# Test 1: Simple HTML
st.markdown("### Test 1: Simple HTML")
st.markdown("""
<div style="background: #00ffff; color: #000; padding: 20px; border-radius: 10px;">
    <h2>This should be cyan background with black text</h2>
    <p>If you see this styled properly, HTML rendering is working!</p>
</div>
""", unsafe_allow_html=True)

# Test 2: Complex HTML
st.markdown("### Test 2: Complex HTML")
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; border: 1px solid rgba(0,255,255,0.3);">
    <h2 style="color: #00ffff;">🛡️ NeuralNet Watchdog</h2>
    <p style="color: #b8b8b8;">Advanced AI-Powered Network Anomaly Detection System</p>
    <ul style="color: #b8b8b8;">
        <li>Real-time Analysis</li>
        <li>Multi-threat Detection</li>
        <li>Advanced ML Models</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Test 3: Raw text (should show as code)
st.markdown("### Test 3: Raw Text (should show as code)")
st.code("""
<div style="background: #00ffff; color: #000; padding: 20px;">
    This should show as raw HTML code
</div>
""")

st.success("✅ If you see styled HTML above, the rendering is working correctly!")
st.error("❌ If you see raw HTML code instead of styled content, there's a rendering issue.") 