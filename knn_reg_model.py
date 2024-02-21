import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import time as t

data = pd.read_csv(r'C:\Users\LENOVO\Desktop\StreamLit Project\Student_Prep\student_exam_data.csv')
X = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.55,random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=25)
classifier.fit(X_train,y_train)

import streamlit as st

st.sidebar.image(r"exam.jpg")
st.markdown("<h1 style='color: yellowgreen;'>Evaluate your level of preparation across students</h1>", unsafe_allow_html=True)

st.sidebar.title("Hello, my fellow students and peers")
choice = st.sidebar.radio("What brings you here today : ",["See students data","Predict my exam prep","Contribute to the dataset"])


if choice == "See students data":
    fig, ax = plt.subplots(figsize=(8, 6))
    st.header("Students past records")

    lower = st.slider("Enter the lower index",0,500)
    upper = st.slider("Enter the upper index",0,500)
    
    plt.scatter(X[lower:upper+1, 0], X[lower:upper+1, 1],color="red")
    plt.xlabel('Hours studied')
    plt.ylabel('Previous marks')
    plt.title('Hours studied vs Marks obtained graph')
    plt.grid()
    st.pyplot(fig)


elif choice == "Predict my exam prep":
    st.header("Hold on champ....All the best")
    hours = st.slider("Enter the hours you studied for the exam",0.0,100.0,step=0.25)
    prev_score = st.slider("Enter the score of previous exam",0.0,100.0,step=0.25)
    
    btn = st.button("Predict : Pass v/s Fail")
    if(btn):
        
        with st.spinner("Hold on...computing your future"):
            t.sleep(3)
        
        test_value = sc.transform([[hours,prev_score]])
        result = classifier.predict(test_value)[0]
        if result == 0:
            st.snow()
            t.sleep(1)
            st.error("Champ...work harder or you will fail")
        else:
            st.balloons()
            st.success("Great champ.... you will pass the exam with flying colors")

else:
    st.header("Thank you for your contribution...")

    hours_contri = st.slider("Enter the hours you studied for the exam",0.0,100.0,step=0.25)
    prev_score_contri = st.slider("Enter the score of previous exam",0.0,100.0,step=0.25)
    result_contri = st.slider("Did you pass the exam : ",0,1,step=1)

    st.markdown("<h5 style='color: lightblue;'>Enter : 0 - Fail, 1 - Pass</h5>", unsafe_allow_html=True)

    contri = st.button("Contribute your data...")

    if(contri):
        with st.spinner("Processing your data"):
            t.sleep(3)
        contri_data = pd.DataFrame([[hours_contri,prev_score_contri,result_contri]])
        contri_data.to_csv("temp.csv",mode="a",header=False,index=False)
        st.success("Data stored successfully....")