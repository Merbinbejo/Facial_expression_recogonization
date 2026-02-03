import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

model=YOLO("model.pt")

st.set_page_config(
    page_title="Facial Expression Recognition",
    layout="wide",
    page_icon="download.png"
)

st.markdown("""
<style>
.centre {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction:column;
    background-color:#E8F5E9;
            
    border:2px solid #4CAF50;
    padding:20px;
    border-radius:10px;  
    width:60%;
    margin:auto; 
}
            
.centre h1{
           text-align:center
            width:100% 
            font-family:"Segoe UI", Arial, sans-serif;
            font-size:40px;
            font-weight:700;
            letter-spacing:1px}
</style>
            

""", unsafe_allow_html=True)

st.markdown(
    """<div class='centre'><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQYRr10WjzDxnyCfnqHay6uWGLxQrleJan4dA&s"><h1>Facial Expression Recognition</h1></div>""",
    unsafe_allow_html=True
)

st.divider()

st.markdown("""
<div style="background-color:#E8F5E9;
            border-radius:10px;
            text-align:center;
            margin:auto;
            padding:10px 15px;"><h4>This Streamlit application uses a deep learning model to predict facial expressions from uploaded images. The system analyzes facial features and displays the detected emotion along with bounding boxes for easy visualization.</h4>
    
</div>

""",unsafe_allow_html=True)

tab1,= st.tabs(
    ["Image Prediction".upper()]
)

st.markdown("""

        <style>
            /*container*/
            .stTabs [data-baseweb="tab-list"]{
            display:flex;
            justify-content:center;
            gap:20px;
            margin-top:10px;
            }
        /*individual container*/
            .stTabs [data-baseweb="tab"]{
            height:50px;
            background-color:#E8F5E9;
            border-radius:10px;
            padding:5px 20px;
            font-size:16px;
            font-weight:700;

            }
        /*active tabs*/
            .stTabs [aria-selected="true"]{
            background-color: #4CAF50;
            color:white;
            }
            </style>

""",unsafe_allow_html=True)

with tab1:
    with st.container():
         a,b,c=st.columns(3)
         with b:
            upload_file=st.file_uploader("Upload Image ",type=["jpg","png","jpeg"])

            st.markdown("""
                        <style>
                        /* stbutton*/
                        .stButton button{
                        background-color:#4CAF50;
                        color:white;
                        padding: 10px 25px;
                        font-size:20px;
                        font-weight:700;
                        border-radius:10px;
                        border:none;
                        transition: all 0.3s ease;
                        }

                        /* hover effect */
                        .stButton button:hover {
                            background-color: #45a049; /* Darker green on hover */
                            cursor: pointer;
                        }
                        </style>
            """,unsafe_allow_html=True)
            button1=st.button("Predict")

            if button1:
                if upload_file is not None:
                    st.image(upload_file)
                    img=Image.open(upload_file).convert("RGB")

                    img1=np.array(img)
                    

                    results=model.predict(img1)
                    result=results[0]
                    if len(result.boxes)>0:
                            confs=[float(box.conf[0]) for box in result.boxes]
                            top_id=np.argmax(confs)
                            top_box=result.boxes[top_id]

                            cls_idx=int(top_box.cls[0])
                            conf=float(top_box.conf[0])
                            name=model.names[cls_idx]
                            st.markdown(f"""
                                        <div style="background-color:#4E5b31;
                                            border-radius:12px;
                                            color:white;
                                            text-align:center;
                                            border:1px solid #3d4727;
                                            box-shadow:2px 2px 6px rgba(0,0,0,0.08);
                                            padding:6px 14px;"><h3 style="margin:4px 0;font-size:20px;font-weight:600;">The Image Shows : {name.upper()} expression</h3></div>
        """,unsafe_allow_html=True)

