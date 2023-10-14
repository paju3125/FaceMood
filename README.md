# FaceMood

<p style="text-align: justify">Step into the realm of emotion analysis with FaceMood, an innovative web application meticulously developed to decode facial expressions in real-time. Seamlessly blending Streamlit's user-friendly web interface, the prowess of OpenCV and MediaPipe for live webcam interactions, and the robust intelligence of Keras and TensorFlow for emotion recognition, FaceMood brings forth a captivating experience. Witness the convergence of technology and human emotion as FaceMood unravels the unspoken through expressions – your emotions, our insight.</p>

---
## Installation

download model from [mydrive](https://drive.google.com/file/d/1FO1pJNvD6oIyfskkeTiO_jTcAh_Wc1aP/view?usp=sharing) and store it in models directory.
you can also train your own model in training folder and save that model in models folder.

1. Create a virtual environment
```bash
  python -m venv venv
```

2. Activate environment
```bash
  .\venv\Scripts\activate
```

3. Install dependencies
```bash
  pip install -r requirements.txt
```

4. Run streamlit webapp
```bash
  streamlit run app.py
```
