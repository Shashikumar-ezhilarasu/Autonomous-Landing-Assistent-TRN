# 🚀 Streamlit Cloud Deployment Guide

## OpenCV Import Error Fix

This guide resolves the common OpenCV import error on Streamlit Cloud.

### Files Added/Modified:

1. **requirements.txt** - Updated to use `opencv-python-headless`
2. **packages.txt** - Added system dependencies for OpenCV
3. **app.py** - Added error handling and fallback functions
4. **requirements-streamlit.txt** - Alternative requirements file

### Deployment Steps:

#### Option 1: Use packages.txt (Recommended)
1. Make sure `packages.txt` exists in your app directory
2. Use `opencv-python-headless` in `requirements.txt`
3. Deploy as normal

#### Option 2: Use alternative requirements
1. Rename `requirements-streamlit.txt` to `requirements.txt`
2. Deploy the app

#### Option 3: Manual fixes
1. In Streamlit Cloud settings, add these packages to "Advanced Settings":
   ```
   libgl1-mesa-glx
   libglib2.0-0
   libsm6
   libxext6
   libxrender-dev
   libgomp1
   ```

### Testing Locally:
```bash
# Test with headless OpenCV
pip install opencv-python-headless
streamlit run app.py
```

### Common Issues & Solutions:

1. **Still getting import errors?**
   - Clear Streamlit Cloud cache
   - Reboot the app
   - Check logs for specific missing dependencies

2. **Some features not working?**
   - The app now includes fallback functions
   - Basic functionality will work even without full OpenCV

3. **Performance issues?**
   - Headless OpenCV should be faster
   - Consider reducing image processing intensity

### Support:
If issues persist, check the Streamlit Community Forum or GitHub issues for opencv-python-headless.
