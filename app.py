import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from pycombat import Combat  # 导入 Combat
import os

# 获取当前脚本所在路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
MODEL_PATH = os.path.join(BASE_DIR, "model", "ann_age_prediction_model.keras")

# 加载模型
model = tf.keras.models.load_model(MODEL_PATH)

# 加载标准化参数
x_min = np.load('x_min.npy')
x_max = np.load('x_max.npy')

# Streamlit UI
st.title("SkinAGE")

# 文件上传组件，用户可以上传CSV文件
uploaded_file = st.file_uploader("Upload your gene expression data (Excel .xlsx format)", type=["xlsx"])

if uploaded_file is not None:
    # 读取上传的CSV文件
    input_data = pd.read_excel(uploaded_file)

    # 确保基因名称在第一列，表达量在第二列
    input_data = input_data.set_index(input_data.columns[0])  # 将基因名称设置为索引

    # 获取上传数据中的列名（样本名称），这些列名是第二列开始的
    sample_names = input_data.columns

    # 确保上传的数据包含所有目标基因
    target_genes = [
        "PLSCR4", "KCNK2", "DSEL", "SVEP1", "CDC14B", "TTC39C",
        "SESN1", "CFAP69", "CTSO", "SERPING1", "IGIP", "SLC25A27",
        "PROS1", "RECK", "CYBRD1", "NIPAL2", "IL6ST", "BICC1",
        "ANKRD18B", "ATL1", "CLDND2", "CRACR2A", "INTS4P1", "NEFH",
        "OLFM1", "PCDHGB8P", "PURG", "PYCR1", "RANBP17", "TUBA3FP",
        "XKR6", "ZNF354C"
    ]

    # 检查是否有缺失的目标基因，并填充为0
    missing_genes = [gene for gene in target_genes if gene not in input_data.index]
    for gene in missing_genes:
        input_data.loc[gene] = 0  # 用0填充缺失的基因数据

    # 提取目标基因的数据（保持顺序并确保格式正确）
    input_data = input_data.loc[target_genes].values.T  # 转置，使得每一行是一个样本

    # 假设训练数据已经在之前加载过，并且目标基因数据已合并
    # 模拟加载训练数据（您可以替换为您真实的训练数据加载方式）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TRAIN_PATH = os.path.join(BASE_DIR, "traindata.xlsx")

    train_data = pd.read_excel(TRAIN_PATH)
    train_data = train_data.set_index(train_data.columns[0])  # 将第一列设置为索引
    train_x_data = train_data.loc[target_genes].values.T  # 确保训练数据格式一致

    # 合并训练数据与新数据
    combined_data = np.concatenate([train_x_data, input_data], axis=0)

    # 创建批次信息，训练数据为批次1，输入数据为批次2
    batch_info = np.array([1] * train_x_data.shape[0] + [2] * input_data.shape[0])

    # 初始化Combat对象并去除批次效应
    combat = Combat()
    corrected_data = combat.fit_transform(combined_data, batch_info)

    # 提取去除批次效应后的新数据部分
    corrected_input_data = corrected_data[train_x_data.shape[0]:, :]  # 提取新输入数据的去批次效应后的部分

    # 归一化数据
    corrected_input_data_normalized = (corrected_input_data - x_min) / (x_max - x_min)

    # 显示上传的数据和去批次效应后的数据
    st.write("Uploaded data:")
    st.write(input_data)

    st.write("Batch effect removed data:")
    st.write(corrected_input_data)

    st.write("Normalized data:")
    st.write(corrected_input_data_normalized)

    # 当用户点击“预测”按钮时进行预测
    if st.button("Predict"):
        # 对每个样本进行预测
        predictions = model.predict(corrected_input_data_normalized)

        # 显示每个样本的预测结果
        st.write("Predicted Results:")
        for i, pred in enumerate(predictions):
            # 输出样本名称与预测结果
            st.write(f"'{sample_names[i]}': {pred[0]}")