import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from scipy import stats
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
import warnings
import io
import re
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain.chat_models import ChatDeepSeek
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

warnings.filterwarnings('ignore')
load_dotenv()

st.set_page_config(
    page_title="AI科研数据分析平台",
    page_icon="🤖📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
    <style>
    .stApp {background-color: var(--backgroundColor);font-family: var(--font);}
    .stButton > button {background-color: #1e88e5;border-radius: 8px;border: none;padding: 8px 16px;color: white;transition: all 0.3s;}
    .stButton > button:hover {background-color: #1976d2;box-shadow: 0 4px 8px rgba(0,0,0,0.15);}
    .card {background: white;border-radius: 12px;padding: 16px;margin: 8px 0;box-shadow: 0 1px 3px rgba(0,0,0,0.05);}
    .ai-report {line-height: 1.6;margin: 12px 0;}
    .sidebar-header {font-size: 16px;font-weight: bold;color: #1e88e5;margin: 16px 0 8px 0;}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

@st.cache_data(show_spinner="加载数据中...")
def load_and_clean_data(file):
    encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312']
    seps = [',', '\t', ';']
    try:
        file_content = file.read()
        file.seek(0)
        df = None
        if file.name.endswith(".csv"):
            for encoding in encodings:
                for sep in seps:
                    try:
                        if encoding == 'utf-16':
                            content = file_content.decode(encoding, errors='replace')
                            df = pd.read_csv(io.StringIO(content), sep=sep, on_bad_lines='skip')
                        else:
                            df = pd.read_csv(file, encoding=encoding, sep=sep, on_bad_lines='skip')
                        break
                    except:
                        continue
                if df is not None:
                    break
            if df is None:
                from csv import Sniffer
                sample = file_content[:4096].decode('utf-8-sig', errors='replace')
                delimiter = Sniffer().sniff(sample).delimiter
                df = pd.read_csv(file, encoding='utf-8-sig', sep=delimiter, on_bad_lines='skip')
        else:
            df = pd.read_excel(file, engine='openpyxl')
        df.columns = [re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() for col in df.columns]
        df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
        return df
    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        return None

def identify_variable_types(df):
    numeric_cols = []
    categorical_cols = []
    binary_categorical_cols = []
    datetime_cols = []
    for col in df.columns:
        if any(fmt in col.lower() for fmt in ['date', 'time', '2016', '2017', '2018']):
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
                continue
            except:
                pass
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except:
            categorical_cols.append(col)
            if df[col].nunique() == 2:
                binary_categorical_cols.append(col)
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'binary_categorical': binary_categorical_cols,
        'datetime': datetime_cols
    }

@st.cache_resource(show_spinner="初始化AI引擎...")
def init_ai_agent(df):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("❌ 未配置API密钥，请检查.env文件或部署环境变量")
        return None
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.3
    )
    tool = PythonAstREPLTool(
        locals={"df": df, "pd": pd, "np": np, "plt": plt, "px": px, "alt": alt, "stats": stats},
        description="执行Python数据分析代码，可访问df数据集"
    )
    system_prompt = """
    你是科研数据分析专家，基于df数据集完成专业分析：
    1. 先输出数据概况（规模、变量类型、缺失值）；
    2. 自动识别有价值的分析点（相关性、分组差异、趋势等）；
    3. 用Python生成统计结果和可视化图表（保存为plot.png）；
    4. 结合本科生科研场景解读结果，避免纯技术术语；
    5. 生成结构化结论，含统计学依据（如p值、R²）。
    """
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool], verbose=False, handle_parsing_errors="请生成正确Python代码")

def auto_ai_analysis(df):
    agent_executor = init_ai_agent(df)
    if not agent_executor:
        return "AI初始化失败"
    auto_query = """
    完成：1.数据概况；2.数值变量统计；3.2个以上核心分析；4.1个可视化图表；5.3条科研结论
    """
    with st.spinner("🤖 AI自主分析中..."):
        response = agent_executor.invoke({"input": auto_query, "chat_history": []})
    return response["output"]

st.title("🤖 AI驱动科研数据分析平台")
st.markdown("**低代码操作 · 自然语言交互 · 专业报告生成**")
st.divider()

with st.sidebar:
    st.markdown('<div class="sidebar-header">1. 上传数据文件</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "支持CSV/Excel（可传多个）",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    df = None
    if uploaded_files:
        st.markdown('<div class="sidebar-header">2. 选择分析文件</div>', unsafe_allow_html=True)
        selected_files = st.multiselect(
            "勾选参与分析的文件",
            [f.name for f in uploaded_files],
            default=[uploaded_files[0].name]
        )
        selected_file_objs = [f for f in uploaded_files if f.name in selected_files]
        df_dict = {}
        for file in selected_file_objs:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
        if len(df_dict) >= 2:
            st.markdown('<div class="sidebar-header">3. 多文件关联</div>', unsafe_allow_html=True)
            base_file = st.selectbox("选择基础文件", list(df_dict.keys()))
            df = df_dict[base_file]
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                common_cols = [col for col in df.columns if col in df_other.columns]
                base_key = st.selectbox(
                    f"基础文件关联字段",
                    common_cols if common_cols else df.columns,
                    key=f"base_{other_file}"
                )
                join_key = st.selectbox(
                    f"关联文件关联字段",
                    common_cols if common_cols else df_other.columns,
                    key=f"join_{other_file}"
                )
                if st.button(f"关联[{other_file}]", key=f"btn_{other_file}"):
                    df = pd.merge(
                        df, df_other, left_on=base_key, right_on=join_key,
                        how="left", suffixes=("", f"_{other_file.split('.')[0]}")
                    )
                    st.success(f"✅ 关联后：{len(df)}行 × {len(df.columns)}列")
        else:
            df = df_dict[list(df_dict.keys())[0]]
        if df is not None:
            var_types = identify_variable_types(df)
            st.markdown('<div class="sidebar-header">4. 变量类型</div>', unsafe_allow_html=True)
            st.write(f"📈 数值型：{', '.join(var_types['numeric'][:4])}{'...' if len(var_types['numeric'])>4 else ''}")
            st.write(f"🏷️ 分类型：{', '.join(var_types['categorical'][:4])}{'...' if len(var_types['categorical'])>4 else ''}")
            st.write(f"❌ 缺失值：{df.isnull().sum().sum()}个（{df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.1f}%）")

if df is not None:
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("数据预览（前5行）")
        st.dataframe(df.head(), use_container_width=True, height=220)
    with col2:
        st.subheader("数据概况")
        st.markdown(f"""
        <div class="card">
        <p>📊 规模：{len(df)}行 × {len(df.columns)}列</p>
        <p>📈 数值列：{len(var_types['numeric'])}个</p>
        <p>🏷️ 分类列：{len(var_types['categorical'])}个</p>
        <p>⏰ 时间列：{len(var_types['datetime'])}个</p>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    tab1, tab2 = st.tabs(["🤖 AI自动分析", "💬 自然语言提问"])
    with tab1:
        if "ai_report" not in st.session_state:
            st.session_state["ai_report"] = None
        if st.button("🚀 启动AI分析", type="primary"):
            st.session_state["ai_report"] = auto_ai_analysis(df)
        if st.session_state["ai_report"]:
            st.subheader("📊 AI分析报告")
            st.markdown(f'<div class="ai-report">{st.session_state["ai_report"]}</div>', unsafe_allow_html=True)
            if os.path.exists("plot.png"):
                st.subheader("📈 生成图表")
                st.image("plot.png", use_container_width=True)
                os.remove("plot.png")
    with tab2:
        st.subheader("输入分析需求（示例：分析两种教学方法对成绩的影响）")
        user_query = st.text_area("自然语言描述你的需求", placeholder="1. 分析城市与订单量的相关性\n2. 按性别分组对比分数差异")
        if st.button("提交提问") and user_query:
            agent_executor = init_ai_agent(df)
            if agent_executor:
                with st.spinner("🤖 处理中..."):
                    response = agent_executor.invoke({"input": user_query, "chat_history": []})
                st.subheader("💡 分析结果")
                st.markdown(f'<div class="ai-report">{response["output"]}</div>', unsafe_allow_html=True)
                if os.path.exists("plot.png"):
                    st.image("plot.png", use_container_width=True)
                    os.remove("plot.png")
    if st.session_state.get("ai_report"):
        st.divider()
        report_content = f"""# AI科研数据分析报告
## 分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
## 数据概况：{len(df)}行 × {len(df.columns)}列
## 核心结论：
{st.session_state['ai_report']}
"""
        st.download_button(
            label="📥 下载报告（Markdown）",
            data=report_content,
            file_name=f"AI分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.md",
            mime="text/markdown"
        )
else:
    st.info("💡 请在侧边栏上传数据文件（支持任意CSV/Excel）")
