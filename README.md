# 🛡️ MargRakshak

**MargRakshak** is an AI-powered public infrastructure monitoring solution designed to detect road damage, track potholes and overflowing drains in real-time, and automate municipal work-order dispatching.

## Features
- **Real-Time Computer Vision Feed:** Automated detection of road hazards (potholes, drains, defects).
- **Interactive Geo-Tagging:** Real-time mapping using Folium with severity color-coding.
- **Municipal Work-Order Generation:** Instant PDF dispatch documents for city repair teams.
- **Predictive Heatmap:** Flag high-degradation road segments for preventive maintenance.

## Quick Start
```bash
pip install streamlit ultralytics opencv-python-headless folium reportlab numpy
streamlit run app.py
```
