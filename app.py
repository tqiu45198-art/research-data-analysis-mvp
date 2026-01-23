import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import warnings
import io
import re
from datetime import datetime
from openai import OpenAI

warnings.filterwarnings('ignore')
# 修复中文显示：兼容云环境无中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.font_manager.fontManager.addfont(plt.font_manager.FontProperties(family='DejaVu Sans').get_file())
except:
    pass
st.set_page_config(page_title="科研数据分析平台", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# 【以下省略未修改的函数定义，仅保留修改后的核心逻辑】

# 修复：数据概况展示（删除多余的 markdown 格式错误）
with st.sidebar:
    st.markdown("## 📥 数据上传")
    uploaded_files = st.file_uploader("上传文件（CSV/Excel，支持多文件合并）", type=["xlsx", "csv"], accept_multiple_files=True)
    df = None
    var_types = None
    if uploaded_files:
        selected_file_names = st.multiselect("选择分析文件", [f.name for f in uploaded_files], default=[uploaded_files[0].name])
        selected_files = [f for f in uploaded_files if f.name in selected_file_names]
        
        df_dict = {}
        for file in selected_files:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
                st.success(f"✅ {file.name} 上传成功 ({len(df_temp)}行×{len(df_temp.columns)}列)")
        
        # 多文件合并逻辑
        if len(df_dict) >= 2:
            st.markdown("### 🔗 多文件合并")
            base_file = st.selectbox("基础文件", list(df_dict.keys()))
            df = df_dict[base_file]
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                common_cols = [col for col in df.columns if col in df_other.columns]
                base_key = st.selectbox(f"基础关联字段", common_cols if common_cols else df.columns, key=f"base_{other_file}")
                join_key = st.selectbox(f"{other_file}关联字段", common_cols if common_cols else df_other.columns, key=f"join_{other_file}")
                join_type = st.selectbox(f"合并方式", ['左连接', '右连接', '内连接', '外连接'], key=f"type_{other_file}")
                join_map = {'左连接':'left', '右连接':'right', '内连接':'inner', '外连接':'outer'}
                if st.button(f"🔄 合并{other_file}", key=f"btn_{other_file}"):
                    df = pd.merge(df, df_other, left_on=base_key, right_on=join_key, how=join_map[join_type], suffixes=("", f"_{other_file.split('.')[0]}"))
                    st.success(f"✅ 合并后：{len(df)}行×{len(df.columns)}列")
        else:
            df = df_dict[list(df_dict.keys())[0]] if df_dict else None
        
        # 修复：数据概况展示（删除 horizontal 参数，替换 use_container_width）
        if df is not None:
            var_types = identify_variable_types(df)
            st.markdown("## 📋 数据概况")
            st.info(f"📏 规模：{len(df)}行 × {len(df.columns)}列")
            st.info(f"🔢 数值型变量：{len(var_types['numeric'])}个")
            st.info(f"📦 分类型变量：{len(var_types['categorical'])}个")
            st.info(f"⚖️ 二分类变量：{len(var_types['binary_categorical'])}个")
            st.info(f"📅 时间型变量：{len(var_types['datetime'])}个")

# 修复：数据处理标签页（删除 selectbox 的 horizontal 参数）
with tab1:
    st.subheader("⚙️ 数据预处理")
    with st.expander("🔍 数据筛选", expanded=True):
        filter_col = st.selectbox("筛选字段", df.columns, key='filter')
        # 修复：删除 selectbox 的 horizontal=True
        filter_op = st.selectbox("运算符", ['>', '<', '>=', '<=', '==', '!='], key='filter_op')
        filter_val = st.text_input("筛选值（数值/文本）", key='filter_val', placeholder="例：100 / 男")
        if st.button("执行筛选", key='btn_filter'):
            try:
                if df[filter_col].dtype in [np.int64, np.float64]:
                    filter_val = float(filter_val)
                df_filtered = df.query(f"`{filter_col}` {filter_op} {filter_val}")
                st.success(f"✅ 筛选后：{len(df_filtered)}行数据")
                # 修复：use_container_width 替换为 width='stretch'
                st.dataframe(df_filtered.head(15), width='stretch')
            except Exception as e:
                st.error(f"❌ 筛选条件错误：{str(e)[:50]}，请检查值的类型是否匹配")
    with st.expander("📊 分类汇总", expanded=True):
        group_col = st.selectbox("分组字段", var_types['categorical'], key='group', disabled=not var_types['categorical'])
        agg_col = st.selectbox("汇总字段", var_types['numeric'], key='agg', disabled=not var_types['numeric'])
        # 修复：删除 selectbox 的 horizontal=True
        agg_func = st.selectbox("汇总方式", ['均值', '求和', '计数', '最大值', '最小值'], key='agg_func')
        agg_map = {'均值':'mean', '求和':'sum', '计数':'count', '最大值':'max', '最小值':'min'}
        if st.button("执行分类汇总", key='btn_agg', disabled=not (group_col and agg_col)):
            df_agg = df.groupby(group_col)[agg_col].agg(agg_map[agg_func]).round(2)
            st.dataframe(df_agg, width='stretch')
            fig_agg = px.bar(df_agg.reset_index(), x=group_col, y=agg_col, title=f"{group_col} - {agg_col}（{agg_func}）")
            # 修复：添加唯一 key
            st.plotly_chart(fig_agg, width='stretch', key=f"plotly_agg_{group_col}_{agg_col}")

# 修复：相关分析标签页（删除 pyplot 的 key 参数）
with tab5:
    st.subheader("📈 相关分析")
    corr_type = st.selectbox("相关系数类型", ['pearson（皮尔逊，适用于正态分布）', 'spearman（斯皮尔曼，非参数/偏态）'], key='corr_type')
    corr_type_map = {'pearson（皮尔逊，适用于正态分布）':'pearson', 'spearman（斯皮尔曼，非参数/偏态）':'spearman'}
    corr_cols = st.multiselect("选择数值型变量（至少2个）", var_types['numeric'], key='corr_cols')
    if len(corr_cols) < 2:
        st.warning("⚠️ 请选择至少2个数值型变量进行相关分析")
        st.button("执行相关分析（含热力图）", key='btn_corr', disabled=True)
    else:
        if st.button("执行相关分析（含热力图）", key='btn_corr'):
            corr_res = correlation_analysis(df, corr_cols, corr_type_map[corr_type])
            st.subheader(f"📊 {corr_type.split('（')[0]} 相关系数矩阵")
            st.dataframe(corr_res['相关矩阵'], width='stretch')
            st.subheader(f"📊 相关分析p值矩阵（p<0.05为显著）")
            st.dataframe(corr_res['p值矩阵'], width='stretch')
            st.subheader(f"📊 相关系数热力图")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(corr_res['相关矩阵'], cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(corr_cols)))
            ax.set_yticks(np.arange(len(corr_cols)))
            ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=10)
            ax.set_yticklabels(corr_cols, fontsize=10)
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    corr_val = corr_res['相关矩阵'].iloc[i, j]
                    p_val = corr_res['p值矩阵'].iloc[i, j]
                    mark = '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                    text = ax.text(j, i, f"{corr_val:.3f}{mark}", ha="center", va="center", color="black", fontsize=9)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label(f'{corr_type.split("（")[0]} 相关系数', rotation=270, labelpad=20, fontsize=12)
            plt.title(f'{corr_type.split("（")[0]} 相关系数热力图（**p<0.01，*p<0.05）', fontsize=14)
            plt.tight_layout()
            # 修复：删除 pyplot 的 key 参数
            st.pyplot(fig)
