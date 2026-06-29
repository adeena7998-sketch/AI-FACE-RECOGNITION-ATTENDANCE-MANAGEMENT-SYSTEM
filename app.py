import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

logo = Image.open("logo.png")

st.set_page_config(
    page_title="AI Face Recognition Attendance Management System",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PROFESSIONAL CSS
# ==========================================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.block-container{
padding-top:1rem;
padding-left:2rem;
padding-right:2rem;
padding-bottom:1rem;
background:#f4f7fc;
}

section[data-testid="stSidebar"]{
background:#0F172A;
}

section[data-testid="stSidebar"] *{
color:white;
}

.main-title{
font-size:38px;
font-weight:700;
color:#0F172A;
margin-bottom:0px;
}

.subtitle{
font-size:17px;
color:#64748b;
margin-bottom:20px;
}

.card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0 6px 20px rgba(0,0,0,.08);
margin-bottom:20px;
}

.metric-card{
background:white;
padding:20px;
border-radius:18px;
text-align:center;
box-shadow:0 6px 20px rgba(0,0,0,.08);
}

.metric-number{
font-size:34px;
font-weight:bold;
color:#2563eb;
}

.metric-text{
font-size:15px;
color:#666;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

left,right = st.columns([1,8])

with left:
    st.image(logo,width=90)

with right:
    st.markdown(
        "<div class='main-title'>AI Face Recognition Attendance Management System</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Smart Face Recognition Based Attendance System</div>",
        unsafe_allow_html=True
    )

st.divider()

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "keras_model.h5",
        compile=False
    )

model = load_model()

# ==========================================
# LOAD LABELS
# ==========================================

with open("labels.txt") as f:

    class_names=[]

    for line in f.readlines():

        line=line.strip()

        parts=line.split(" ",2)

        if len(parts)>1:
            class_names.append(parts[1])

        else:
            class_names.append(parts[0])

# ==========================================
# CREATE CSV
# ==========================================

if not os.path.exists("attendance.csv"):

    pd.DataFrame(
        columns=[
            "Sr No",
            "ID",
            "Name",
            "Date",
            "Time",
            "Status"
        ]
    ).to_csv(
        "attendance.csv",
        index=False
    )

# ==========================================
# IMAGE PREDICTION
# ==========================================

def predict_image(image):

    image=image.convert("RGB")

    image=image.resize((224,224))

    image_array=np.asarray(image)

    normalized=(image_array.astype(np.float32)/127.5)-1

    data=np.ndarray(
        shape=(1,224,224,3),
        dtype=np.float32
    )

    data[0]=normalized

    prediction=model.predict(data,verbose=0)

    index=np.argmax(prediction)

    confidence=float(prediction[0][index])*100

    return class_names[index],confidence

# ==========================================
# MARK ATTENDANCE
# ==========================================

def mark_attendance(name):

    df=pd.read_csv("attendance.csv")

    today=datetime.now().strftime("%Y-%m-%d")

    already=df[
        (df["Name"]==name)&
        (df["Date"]==today)
    ]

    if len(already)>0:

        return False

    sr=len(df)+1

    ai_id=f"AI-{sr}"

    new_row={
        "Sr No":sr,
        "ID":ai_id,
        "Name":name,
        "Date":today,
        "Time":datetime.now().strftime("%H:%M:%S"),
        "Status":"Present"
    }

    df=pd.concat(
        [df,pd.DataFrame([new_row])],
        ignore_index=True
    )

    df.to_csv(
        "attendance.csv",
        index=False
    )

    return True

# ==========================================
# LOAD DATA
# ==========================================

attendance=pd.read_csv("attendance.csv")

today=datetime.now().strftime("%Y-%m-%d")

today_df=attendance[
    attendance["Date"]==today
]

total_students=len(class_names)

present_today=len(today_df)

absent_today=max(
    total_students-present_today,
    0
)

attendance_percent=0

if total_students>0:

    attendance_percent=(
        present_today/total_students
    )*100

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.image(
    logo,
    width=150
)

st.sidebar.markdown("# AI Attendance")

st.sidebar.markdown("---")

menu=st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📷 Mark Attendance",

        "📋 Attendance Records",

        "📅 Daily Summary",

        "📊 Reports",

        "👥 Student Database",

        "⚙ Settings",

        "👨‍💼 Admin",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("System Online")

st.sidebar.write(
    datetime.now().strftime("%d %B %Y")
)

st.sidebar.write(
    datetime.now().strftime("%I:%M %p")
)
# ==========================================
# DASHBOARD
# ==========================================

if menu == "🏠 Dashboard":

    st.markdown("## 📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_students}</div>
            <div class="metric-text">Registered Students</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{present_today}</div>
            <div class="metric-text">Present Today</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{absent_today}</div>
            <div class="metric-text">Absent Today</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{attendance_percent:.1f}%</div>
            <div class="metric-text">Attendance Rate</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Today's Attendance")

    if today_df.empty:
        st.info("No attendance marked today.")
    else:
        st.dataframe(
            today_df,
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# MARK ATTENDANCE
# ==========================================

if menu == "📷 Mark Attendance":

    st.markdown("## 📷 Mark Attendance")

    st.info("Capture an image from Webcam or Upload an Image.")

    st.write("")

    # ======================================
    # WEBCAM FIRST
    # ======================================

    st.subheader("📸 Webcam")

    camera = st.camera_input(
        "Capture Face"
    )

    if camera is not None:

        image = Image.open(camera)

        st.image(
            image,
            width=350
        )

        name, confidence = predict_image(image)

        st.success(f"Detected : {name}")

        st.info(
            f"Confidence : {confidence:.2f}%"
        )

        if st.button(
            "✅ Mark Attendance from Webcam",
            use_container_width=True
        ):

            if mark_attendance(name):

                st.success(
                    "Attendance Marked Successfully."
                )

                st.rerun()

            else:

                st.warning(
                    "Attendance Already Marked Today."
                )

    st.divider()

    # ======================================
    # UPLOAD IMAGE SECOND
    # ======================================

    st.subheader("📁 Upload Image")

    uploaded = st.file_uploader(
        "Choose Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded is not None:

        image = Image.open(uploaded)

        st.image(
            image,
            width=350
        )

        name, confidence = predict_image(image)

        st.success(
            f"Detected : {name}"
        )

        st.info(
            f"Confidence : {confidence:.2f}%"
        )

        if st.button(
            "📋 Mark Attendance from Upload",
            use_container_width=True
        ):

            if mark_attendance(name):

                st.success(
                    "Attendance Marked Successfully."
                )

                st.rerun()

            else:

                st.warning(
                    "Attendance Already Marked Today."
                )

    st.divider()

    st.subheader("Today's Attendance")

    today_df = pd.read_csv("attendance.csv")

    today_df = today_df[
        today_df["Date"] ==
        datetime.now().strftime("%Y-%m-%d")
    ]

    if today_df.empty:

        st.info("No attendance has been marked today.")

    else:

        st.dataframe(
            today_df,
            use_container_width=True,
            hide_index=True
        )
        # ==========================================
# ATTENDANCE RECORDS
# ==========================================

if menu == "📋 Attendance Records":

    st.markdown("## 📋 Attendance Records")

    df = pd.read_csv("attendance.csv")

    col1, col2, col3 = st.columns([2,2,1])

    with col1:

        search = st.text_input(
            "🔍 Search Student",
            placeholder="Search by Name or AI ID..."
        )

    with col2:

        selected_date = st.date_input(
            "📅 Select Date",
            value=datetime.now().date()
        )

    with col3:

        st.metric(
            "Total Records",
            len(df)
        )

    filtered = df.copy()

    # -------------------------------
    # SEARCH FILTER
    # -------------------------------

    if search:

        filtered = filtered[
            filtered["Name"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
            |
            filtered["ID"].astype(str).str.contains(
                search,
                case=False,
                na=False
            )
        ]

    # -------------------------------
    # DATE FILTER
    # -------------------------------

    filtered = filtered[
        filtered["Date"] ==
        str(selected_date)
    ]

    st.write("")

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    a,b,c = st.columns(3)

    with a:

        st.download_button(

            "📥 Download CSV",

            filtered.to_csv(index=False),

            file_name="attendance.csv",

            mime="text/csv",

            use_container_width=True

        )

    with b:

        if st.button(
            "🔄 Refresh",
            use_container_width=True
        ):
            st.rerun()

    with c:

        if st.button(
            "🗑 Clear Records",
            use_container_width=True
        ):

            empty = pd.DataFrame(
                columns=[
                    "Sr No",
                    "ID",
                    "Name",
                    "Date",
                    "Time",
                    "Status"
                ]
            )

            empty.to_csv(
                "attendance.csv",
                index=False
            )

            st.success(
                "Attendance Records Cleared."
            )

            st.rerun()


# ==========================================
# DAILY SUMMARY
# ==========================================

if menu == "📅 Daily Summary":

    st.markdown("## 📅 Daily Attendance Summary")

    df = pd.read_csv("attendance.csv")

    today = datetime.now().strftime("%Y-%m-%d")

    today_df = df[
        df["Date"] == today
    ]

    c1,c2,c3 = st.columns(3)

    with c1:

        st.metric(
            "Present Today",
            len(today_df)
        )

    with c2:

        st.metric(
            "Registered Students",
            total_students
        )

    with c3:

        if total_students == 0:

            percent = 0

        else:

            percent = (
                len(today_df) /
                total_students
            ) * 100

        st.metric(
            "Attendance %",
            f"{percent:.1f}%"
        )

    st.progress(
        min(percent/100,1.0)
    )

    st.write("")

    st.dataframe(
        today_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("")

    st.success(
        f"{len(today_df)} students are present today."
    )
    # ==========================================
# REPORTS & ANALYTICS
# ==========================================

if menu == "📊 Reports":

    st.markdown("## 📊 Reports & Analytics")

    df = pd.read_csv("attendance.csv")

    if df.empty:

        st.warning("No attendance records found.")

    else:

        total_records = len(df)

        total_present = len(df[df["Status"] == "Present"])

        attendance_rate = (
            total_present / total_records
        ) * 100

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", total_records)

        with col2:
            st.metric("Present", total_present)

        with col3:
            st.metric(
                "Attendance %",
                f"{attendance_rate:.2f}%"
            )

        st.write("")

        st.subheader("Attendance Progress")

        st.progress(attendance_rate / 100)

        st.write(f"{attendance_rate:.2f}% Attendance")

        st.divider()

        left, right = st.columns(2)

        with left:

            fig, ax = plt.subplots(figsize=(5,5))

            ax.pie(
                [
                    total_present,
                    total_records-total_present
                ],
                labels=[
                    "Present",
                    "Absent"
                ],
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title("Attendance Distribution")

            st.pyplot(fig)

        with right:

            daily = (
                df.groupby("Date")
                .size()
                .reset_index(name="Attendance")
            )

            fig2, ax2 = plt.subplots(figsize=(6,4))

            ax2.bar(
                daily["Date"],
                daily["Attendance"]
            )

            ax2.set_xlabel("Date")

            ax2.set_ylabel("Students")

            ax2.set_title("Daily Attendance")

            plt.xticks(rotation=35)

            st.pyplot(fig2)

        st.divider()

        st.subheader("Latest Attendance Records")

        st.dataframe(
            df.tail(15),
            use_container_width=True,
            hide_index=True
        )

# ==========================================
# STUDENT DATABASE
# ==========================================

if menu == "👥 Student Database":

    st.markdown("## 👥 Registered Students")

    students = pd.DataFrame({

        "ID":[
            f"AI-{i+1}"
            for i in range(len(class_names))
        ],

        "Student Name":class_names

    })

    st.dataframe(

        students,

        use_container_width=True,

        hide_index=True

    )

    st.download_button(

        "📥 Download Student List",

        students.to_csv(index=False),

        "students.csv",

        "text/csv",

        use_container_width=True

    )

# ==========================================
# SETTINGS
# ==========================================

if menu == "⚙ Settings":

    st.markdown("## ⚙ System Settings")

    dark = st.toggle(
        "Dark Theme",
        value=True
    )

    notify = st.toggle(
        "Enable Notifications",
        value=True
    )

    refresh = st.toggle(
        "Auto Refresh",
        value=True
    )

    threshold = st.slider(

        "Recognition Confidence",

        50,

        100,

        80

    )

    st.success(
        f"Recognition Threshold : {threshold}%"
    )

# ==========================================
# ADMIN PANEL
# ==========================================

if menu == "👨‍💼 Admin":

    st.markdown("## 👨‍💼 Administrator Login")

    username = st.text_input("Username")

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button(
        "Login",
        use_container_width=True
    ):

        if username == "admin" and password == "admin123":

            st.success("Administrator Login Successful")

            df = pd.read_csv("attendance.csv")

            st.dataframe(

                df,

                use_container_width=True,

                hide_index=True

            )

            st.download_button(

                "📥 Download Attendance Report",

                df.to_csv(index=False),

                "attendance.csv",

                "text/csv",

                use_container_width=True

            )

        else:

            st.error(
                "Invalid Username or Password."
            )

# ==========================================
# ABOUT
# ==========================================

if menu == "ℹ About":

    st.markdown("## ℹ About")

    st.info("""

### AI Face Recognition Attendance Management System

Developed using

✅ Python

✅ Streamlit

✅ TensorFlow

✅ NumPy

✅ Pandas

✅ Pillow

---

### Features

✔ Live Face Recognition

✔ Webcam Attendance

✔ Upload Image Attendance

✔ AI Generated Student IDs

✔ Dashboard

✔ Attendance Reports

✔ Daily Summary

✔ Student Database

✔ Admin Panel

✔ CSV Download

✔ GitHub Ready

✔ Streamlit Cloud Ready

""")

    st.success(
        "Final Year Project"
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(

"""

<div style="text-align:center;
padding:15px;
color:gray;
font-size:15px;">

AI Face Recognition Attendance Management System

<br><br>

Developed using Python | TensorFlow | Streamlit

</div>

""",

unsafe_allow_html=True

)
