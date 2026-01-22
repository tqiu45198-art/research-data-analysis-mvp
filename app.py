import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import io
import re
import os
from datetime import datetime
from dotenv import load_dotenv  # 加载API密钥

# AI相关导入
from langchain.chat_models import ChatDeepSeek
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

warnings.filterwarnings('ignore')
load_dotenv()  # 加载.env文件中的API密钥

# ---------------------- 1. 页面样式配置 ----------------------
st.set_page_config(
    page_title="AI驱动科研数据分析助手",
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

# ---------------------- 2. 工具函数（数据加载+AI初始化） ----------------------
@st.cache_data(show_spinner="加载数据中...")
def load_and_clean_data(file):
    """加载任意CSV/Excel文件，自动清理列名"""
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
        
        # 清理列名
        df.columns = [re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() for col in df.columns]
        df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
        return df
    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        return None

@st.cache_resource(show_spinner="初始化AI分析引擎...")
def init_ai_agent(df):
    """初始化AI分析代理（大模型+代码执行工具）"""
    # 1. 加载大模型（DeepSeek，需在.env文件中配置DEEPSEEK_API_KEY）
    api_key = os.getenv("sk-158fac228f8b4ee0a06f8ca81013e5fa")
    if not api_key:
        st.error("❌ 请在.env文件中配置DEEPSEEK_API_KEY（免费申请：https://www.deepseek.com/）")
        return None
    
    llm = ChatDeepSeek(
        model="deepseek-chat",
        api_key=api_key,
        temperature=0.3  # 低温度：分析更严谨，减少随机性
    )
    
    # 2. 代码执行工具（仅允许操作df数据，限制风险）
    tool = PythonAstREPLTool(
        locals={"df": df, "pd": pd, "np": np, "plt": plt, "px": px, "alt": alt},
        description="用于执行Python代码分析数据（如统计计算、绘图），可访问df变量（当前数据集）"
    )
    tools = [tool]
    
    # 3. AI提示词（引导AI自主分析数据）
    system_prompt = """
    你是专业的科研数据分析助手，需基于用户上传的数据集（变量df）完成自主分析，遵循以下规则：
    1. 先自动探索数据：输出数据规模（行数/列数）、变量类型（数值型/分类型）、缺失值情况、核心统计指标（均值/标准差/中位数）；
    2. 识别关键问题：自动检测异常值、变量相关性、分组差异等有价值的分析点；
    3. 生成可执行代码：用Python分析（优先用pandas统计、matplotlib/plotly绘图），图表保存为'plot.png'；
    4. 输出专业解读：结合科研场景解释结果（如“p<0.05说明两组差异显著”），避免纯技术术语；
    5. 若数据包含时间/地理信息，需额外分析趋势/分布；
    6. 代码仅操作df变量，禁止修改文件系统（除保存图表）、导入危险库。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # 4. 创建AI代理
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # 关闭详细日志，避免干扰用户
        handle_parsing_errors="请重新生成可执行的Python代码，确保语法正确"
    )
    return agent_executor

def auto_ai_analysis(df):
    """AI自动分析数据的入口函数"""
    agent_executor = init_ai_agent(df)
    if not agent_executor:
        return "AI初始化失败，请检查API密钥"
    
    # AI自动分析的初始指令（无需用户输入，AI自主探索）
    auto_query = """
    基于当前数据集df，完成以下分析：
    1. 数据概况（规模、变量类型、缺失值）；
    2. 数值型变量的核心统计（均值、标准差、异常值）；
    3. 2个以上关键分析（如相关性、分组差异、趋势）；
    4. 生成至少1个可视化图表；
    5. 总结3条以上科研价值结论。
    """
    
    # 执行AI分析
    with st.spinner("🤖 AI正在自主分析数据...（约30-60秒）"):
        response = agent_executor.invoke({"input": auto_query, "chat_history": []})
    return response["output"]

# ---------------------- 3. 核心页面逻辑 ----------------------
st.title("🤖  AI驱动科研数据分析助手")
st.markdown("**数据上传→AI自动分析→生成专业报告**")
st.divider()

# 侧边栏：文件上传
with st.sidebar:
    st.markdown('<div class="sidebar-header">1. 上传数据文件</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "支持CSV/Excel（可传多个）",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    st.markdown('<div class="hint-text">示例：订单数据、实验数据、问卷数据等</div>', unsafe_allow_html=True)
    
    # 加载数据并合并（支持多文件关联）
    df = None
    if uploaded_files:
        # 选择要分析的文件
        st.markdown('<div class="sidebar-header">2. 选择分析文件</div>', unsafe_allow_html=True)
        selected_files = st.multiselect(
            "勾选参与分析的文件",
            [f.name for f in uploaded_files],
            default=[uploaded_files[0].name]
        )
        selected_file_objs = [f for f in uploaded_files if f.name in selected_files]
        
        # 加载文件到字典
        df_dict = {}
        for file in selected_file_objs:
            df_temp = load_and_clean_data(file)
            if df_temp is not None:
                df_dict[file.name] = df_temp
        
        # 多文件关联（按共同字段合并）
        if len(df_dict) >= 2:
            st.markdown('<div class="sidebar-header">3. 多文件关联</div>', unsafe_allow_html=True)
            base_file = st.selectbox("选择基础文件", list(df_dict.keys()))
            df = df_dict[base_file]
            
            for other_file in [f for f in df_dict.keys() if f != base_file]:
                df_other = df_dict[other_file]
                # 自动推荐关联字段（名称/城市等）
                common_cols = [col for col in df.columns if col in df_other.columns]
                base_key = st.selectbox(
                    f"基础文件[{base_file}]关联字段",
                    common_cols if common_cols else df.columns,
                    key=f"base_key_{other_file}"
                )
                join_key = st.selectbox(
                    f"关联文件[{other_file}]关联字段",
                    common_cols if common_cols else df_other.columns,
                    key=f"join_key_{other_file}"
                )
                
                if st.button(f"关联[{other_file}]", key=f"join_btn_{other_file}"):
                    df = pd.merge(
                        df, df_other,
                        left_on=base_key, right_on=join_key,
                        how="left", suffixes=("", f"_{other_file.split('.')[0]}")
                    )
                    st.success(f"✅ 已关联[{other_file}]，当前数据：{len(df)}行 × {len(df.columns)}列")
        else:
            # 单文件分析
            df = df_dict[list(df_dict.keys())[0]]
        
        # 显示数据概况
        if df is not None:
            st.markdown('<div class="sidebar-header">4. 数据概况</div>', unsafe_allow_html=True)
            st.write(f"📊 规模：{len(df)}行 × {len(df.columns)}列")
            st.write(f"❌ 缺失值：{df.isnull().sum().sum()}个（{df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.1f}%）")

# 主页面：数据预览与AI分析
if df is not None:
    # 数据预览（分栏展示）
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("数据预览（前5行）")
        st.dataframe(df.head(), use_container_width=True, height=220)
    
    with col2:
        st.subheader("变量类型识别")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        st.markdown(f"""
        <div class="card">
        <p>📈 数值型变量：{', '.join(numeric_cols[:4])}{'...' if len(numeric_cols)>4 else ''}</p>
        <p>🏷️ 分类型变量：{', '.join(categorical_cols[:4])}{'...' if len(categorical_cols)>4 else ''}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 核心功能：AI自动分析 + 手动提问
    tab1, tab2 = st.tabs(["🤖 AI自动分析", "💬 手动深度提问"])
    
    # Tab1：AI自动分析
    with tab1:
        if "ai_report" not in st.session_state:
            st.session_state["ai_report"] = None
        
        # 触发AI分析
        if st.button("🚀 启动AI自动分析", type="primary"):
            st.session_state["ai_report"] = auto_ai_analysis(df)
        
        # 显示AI分析结果
        if st.session_state["ai_report"]:
            st.subheader("📊 AI分析报告")
            st.markdown(f'<div class="ai-report">{st.session_state["ai_report"]}</div>', unsafe_allow_html=True)
            
            # 显示AI生成的图表（若有）
            if os.path.exists("plot.png"):
                st.subheader("📈 AI生成图表")
                st.image("plot.png", use_container_width=True)
                os.remove("plot.png")  # 避免缓存旧图
    
    # Tab2：手动深度提问（基于已有数据）
    with tab2:
        st.subheader("基于当前数据提问（示例：分析性别对成绩的影响）")
        user_query = st.text_area(
            "输入你的分析需求",
            placeholder="1. 分析两组变量的相关性\n2. 按城市分组对比订单量\n3. 检测异常值并给出处理建议"
        )
        
        if st.button("提交提问") and user_query:
            agent_executor = init_ai_agent(df)
            if agent_executor:
                with st.spinner("🤖 AI正在处理你的问题..."):
                    response = agent_executor.invoke({"input": user_query, "chat_history": []})
                st.subheader("💡 AI回答")
                st.markdown(f'<div class="ai-report">{response["output"]}</div>', unsafe_allow_html=True)
                
                # 显示图表
                if os.path.exists("plot.png"):
                    st.image("plot.png", use_container_width=True)
                    os.remove("plot.png")
    
    # 报告下载
    if st.session_state.get("ai_report"):
        st.divider()
        report_content = f"""# AI科研数据分析报告
## 分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
## 数据概况：{len(df)}行 × {len(df.columns)}列
## AI分析结论：
{st.session_state['ai_report']}
"""
        st.download_button(
            label="📥 下载分析报告（Markdown）",
            data=report_content,
            file_name=f"AI分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.md",
            mime="text/markdown"
        )

# 无数据时的提示
else:
    st.info("💡 请在侧边栏上传数据文件，支持任意结构的CSV/Excel（如实验数据、问卷数据）")
