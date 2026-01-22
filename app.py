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
import warnings
import io
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# LangChain 相关导入（适配最新版本）
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
from langchain_ollama import ChatOllama  # 本地Ollama支持
from langchain_huggingface import HuggingFaceEndpoint  # HuggingFace云端支持

warnings.filterwarnings('ignore')
load_dotenv()

# 页面基础配置
st.set_page_config(
    page_title="AI科研数据分析平台（认知+调度版）",
    page_icon="🔬📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- 1. 核心工具函数（数据加载+变量识别）----------------------
@st.cache_data(show_spinner="加载数据中...")
def load_and_clean_data(file):
    """加载CSV/Excel文件，自动处理编码和分隔符，清理列名"""
    encodings = ['utf-8-sig', 'gbk', 'utf-8', 'gb2312']
    seps = [',', '\t', ';']
    try:
        file_content = file.read()
        file.seek(0)
        df = None
        
        # 处理CSV文件
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
            # 兜底：自动检测分隔符
            if df is None:
                from csv import Sniffer
                sample = file_content[:4096].decode('utf-8-sig', errors='replace')
                delimiter = Sniffer().sniff(sample).delimiter
                df = pd.read_csv(file, encoding='utf-8-sig', sep=delimiter, on_bad_lines='skip')
        # 处理Excel文件
        else:
            df = pd.read_excel(file, engine='openpyxl')
        
        # 清理列名（移除特殊字符，补充空列名）
        df.columns = [re.sub(r'[^\w\s\u4e00-\u9fa5/]', '', str(col)).strip() for col in df.columns]
        df.columns = [col if col else f"col_{i}" for i, col in enumerate(df.columns)]
        return df
    except Exception as e:
        st.error(f"文件读取失败：{str(e)}")
        return None

def identify_variable_types(df):
    """自动识别变量类型：数值型、分类型、二分类、时间型"""
    numeric_cols = []
    categorical_cols = []
    binary_categorical_cols = []
    datetime_cols = []
    
    for col in df.columns:
        # 优先识别时间类变量（含日期/时间关键词或年份）
        if any(fmt in col.lower() for fmt in ['date', 'time', '2016', '2017', '2018', '2019', '2020']):
            try:
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)
                continue
            except:
                pass
        
        # 识别数值型变量
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        # 识别分类型变量
        except:
            categorical_cols.append(col)
            # 二分类变量（唯一值数量=2）
            if df[col].nunique() == 2:
                binary_categorical_cols.append(col)
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'binary_categorical': binary_categorical_cols,
        'datetime': datetime_cols
    }

# ---------------------- 2. LLM模型配置（本地Ollama/云端HuggingFace二选一）----------------------
# 选项1：本地Ollama（推荐，低延迟+数据隐私）
llm = ChatOllama(
    model="llama3-8b-scientific",  # 替换为你的微调模型Tag（如llama3-8b-lora-research）
    temperature=0.25,  # 低温度：科研分析更严谨
    base_url="http://localhost:11434",  # 本地默认地址，远程部署需修改
    max_tokens=2048  # 最大输出长度
)

# 选项2：HuggingFace云端（需配置API Token，适合无本地算力）
# llm = HuggingFaceEndpoint(
#     repo_id="你的用户名/llama3-8b-scientific-lora",  # 你的微调模型仓库ID
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN"),  # 从.env读取Token
#     temperature=0.25,
#     max_new_tokens=2048,
#     model_kwargs={"device": "auto"}  # 自动选择设备（CPU/GPU）
# )

# ---------------------- 3. 科研专用工具链（代码执行+辅助工具）----------------------
@tool
def get_current_time() -> str:
    """返回当前时间，用于报告时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Python代码执行工具（预加载数据分析库，动态注入数据集）
python_repl = PythonAstREPLTool(
    name="python_repl",
    description="""执行Python代码完成科研数据分析，支持：
    1. 统计计算（描述统计、假设检验、相关性分析等）
    2. 可视化（折线图、柱状图、箱线图、热力图等）
    3. 数据预处理（缺失值处理、异常值检测）
    全局变量：
    - df：当前加载的数据集（自动注入）
    - 已导入库：pandas(pd)、numpy(np)、matplotlib(plt)、plotly(px)、scipy(stats)、statsmodels
    绘图要求：保存为plot.png，避免直接plt.show()""",
    globals={"df": None, "pd": pd, "np": np, "plt": plt, "px": px, "alt": alt, "stats": stats}
)

# 工具列表（可扩展：如文献检索工具、统计检验封装工具）
tools = [python_repl, get_current_time]

# ---------------------- 4. ReAct Agent配置（科研场景专属提示词）----------------------
system_prompt = """你是专注于本科生科研的数据分析专家，需严格遵循以下流程完成任务：
1. 意图理解：先明确用户的科研目标（如验证假设、探索变量关系、异常值分析）、核心变量（自变量/因变量）
2. 数据评估：优先用python_repl查看数据概况（行数/列数、变量类型、缺失值），再确定分析方案
3. 工具调用：
   - 统计计算/绘图必须用python_repl，结果需包含统计学指标（如均值、标准差、p值、R²）
   - 绘图需选择科研规范图表（避免花哨样式），保存为plot.png
   - 无需调用工具的简单问题（如方法解释）可直接回答
4. 结果解读：
   - 用"【数据概况】【分析方法】【核心结果】【科研解读与建议】【图表】"结构输出
   - 解释需适配本科生认知（避免过度专业术语，如"p<0.05表示两组差异具有统计学意义"）
   - 给出下一步建议（如"建议补充XX变量的分析""可尝试XX检验验证假设"）
5. 异常处理：数据质量问题（如缺失值>30%）需先提醒用户，再基于可用数据分析"""

# 构建ReAct Prompt模板
prompt = PromptTemplate.from_template(
    system_prompt + "\n\n用户问题：{input}\n\n{agent_scratchpad}"
)

# 创建ReAct Agent与执行器
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # 开发时设为True查看思考过程，生产环境设为False
    handle_parsing_errors="请重新生成符合格式的Python代码，确保语法正确且仅操作df变量",
    max_iterations=15,  # 避免无限循环
    early_stopping_method="generate"
)

# 动态注入数据集到工具
def get_analysis_agent(current_df):
    python_repl.globals["df"] = current_df  # 关键：将当前数据集传入代码执行工具
    return agent_executor

# ---------------------- 5. Streamlit页面核心逻辑（交互+分析）----------------------
st.title("🔬 AI科研数据分析平台（认知+调度分离版）")
st.markdown("**基于微调LLM理解科研意图 | LangChain智能调度分析工具**")
st.divider()

# 侧边栏：文件上传与数据管理
with st.sidebar:
    st.markdown("### 1. 数据上传")
    uploaded_files = st.file_uploader(
        "支持CSV/Excel（可传多个文件关联）",
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    
    df = None  # 全局数据集变量
    if uploaded_files:
        # 步骤1：选择待分析文件
        st.markdown("### 2. 选择分析文件")
        selected_file_names = st.multiselect(
            "勾选需参与分析的文件",
            options=[f.name for f in uploaded_files],
            default=[uploaded_files[0].name]
        )
        selected_files = [f for f in uploaded_files if f.name in selected_file_names]
        
        # 步骤2：加载选中文件到字典
        df_dict = {}
        for file in selected_files:
            file_df = load_and_clean_data(file)
            if file_df is not None:
                df_dict[file.name] = file_df
                st.success(f"✅ 加载成功：{file.name}（{len(file_df)}行×{len(file_df.columns)}列）")
        
        # 步骤3：单文件/多文件关联处理
        if len(df_dict) >= 2:
            st.markdown("### 3. 多文件关联")
            # 选择基础文件
            base_file_name = st.selectbox("选择基础文件", options=list(df_dict.keys()))
            df = df_dict[base_file_name]
            
            # 关联其他文件
            for other_file_name in [f for f in df_dict.keys() if f != base_file_name]:
                other_df = df_dict[other_file_name]
                # 自动识别共同字段
                common_cols = [col for col in df.columns if col in other_df.columns]
                
                st.markdown(f"#### 关联 {other_file_name}")
                base_key = st.selectbox(
                    f"基础文件（{base_file_name}）关联字段",
                    options=common_cols if common_cols else df.columns,
                    key=f"base_key_{other_file_name}"
                )
                other_key = st.selectbox(
                    f"关联文件（{other_file_name}）关联字段",
                    options=common_cols if common_cols else other_df.columns,
                    key=f"other_key_{other_file_name}"
                )
                
                # 执行关联
                if st.button(f"开始关联 {other_file_name}", key=f"join_btn_{other_file_name}"):
                    df = pd.merge(
                        df, other_df,
                        left_on=base_key, right_on=other_key,
                        how="left",  # 左连接：保留基础文件所有数据
                        suffixes=("", f"_{other_file_name.split('.')[0]}")  # 避免列名重复
                    )
                    st.success(f"✅ 关联完成：当前数据（{len(df)}行×{len(df.columns)}列）")
        
        # 单文件直接赋值
        else:
            df = df_dict[list(df_dict.keys())[0]]
        
        # 步骤4：显示数据概况
        if df is not None:
            var_types = identify_variable_types(df)
            st.markdown("### 4. 数据概况")
            st.write(f"📊 数据规模：{len(df)}行 × {len(df.columns)}列")
            st.write(f"📈 数值型变量：{len(var_types['numeric'])}个（{', '.join(var_types['numeric'][:5])}{'...' if len(var_types['numeric'])>5 else ''}）")
            st.write(f"🏷️ 分类型变量：{len(var_types['categorical'])}个（{', '.join(var_types['categorical'][:5])}{'...' if len(var_types['categorical'])>5 else ''}）")
            st.write(f"❌ 缺失值总数：{df.isnull().sum().sum()}个（{df.isnull().sum().sum()/(len(df)*len(df.columns))*100:.1f}%）")

# 主界面：数据预览与AI分析
if df is not None:
    # 数据预览（分栏显示）
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("数据预览（前5行）")
        st.dataframe(df.head(), use_container_width=True, height=220)
    
    with col2:
        st.subheader("变量类型详情")
        var_types = identify_variable_types(df)
        st.markdown(f"""
        <div style="background:white;padding:12px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.05)">
        <p>⏰ 时间型变量：{', '.join(var_types['datetime']) if var_types['datetime'] else '无'}</p>
        <p>🔢 二分类变量：{', '.join(var_types['binary_categorical']) if var_types['binary_categorical'] else '无'}</p>
        <p>⚠️ 高缺失值变量：{[col for col in df.columns if df[col].isnull().sum()/len(df)>0.3] if any(df[col].isnull().sum()/len(df)>0.3 for col in df.columns) else '无'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 分析功能标签页
    tab1, tab2 = st.tabs(["📊 自动科研分析", "💬 自由提问分析"])
    
    # 标签1：自动科研分析（一键生成完整报告）
    with tab1:
        st.subheader("自动科研级探索分析")
        st.markdown("点击按钮后，AI将自动完成：数据质量评估→描述统计→假设检验→可视化→科研解读")
        
        if st.button("🚀 启动自动分析", type="primary", use_container_width=True):
            with st.spinner("🔍 认知层理解科研需求 → 🛠️ 调度层执行分析（约30-60秒）..."):
                # 自动分析指令（明确科研目标）
                auto_query = """
                对当前数据集执行完整探索性科研分析，需包含：
                1. 数据质量评估（缺失值、异常值、变量合理性）
                2. 核心变量描述统计（数值型变量：均值±标准差/中位数；分类型变量：频数+占比）
                3. 2项有科研意义的深度分析（如：数值变量相关性分析+显著性检验、分类型变量组间差异分析（t检验/方差分析））
                4. 1张规范科研图表（如相关性热力图、组间对比箱线图），保存为plot.png
                5. 3-5条本科生可理解的科研结论（含统计学依据）+ 2条下一步研究建议
                """
                # 执行Agent分析
                analysis_agent = get_analysis_agent(df)
                result = analysis_agent.invoke({"input": auto_query})
                
                # 显示分析报告
                st.markdown("### 📋 AI科研分析报告")
                st.markdown(result["output"], unsafe_allow_html=True)
                
                # 显示生成的图表
                if os.path.exists("plot.png"):
                    st.markdown("### 📈 分析图表")
                    st.image("plot.png", use_container_width=True)
                    os.remove("plot.png")  # 清理临时文件，避免缓存
    
    # 标签2：自由提问分析（用户自定义科研需求）
    with tab2:
        st.subheader("基于科研问题的自由分析")
        st.markdown("示例提问：\n1. 分析性别（分类型）对成绩（数值型）的影响，用t检验验证差异显著性\n2. 探索年龄与收入的相关性，绘制散点图并计算Pearson相关系数\n3. 检测销售额的异常值，用箱线图展示并给出处理建议")
        
        user_query = st.text_area(
            "请输入你的科研问题或分析需求",
            height=150,
            placeholder="请详细描述你的分析目标，例如：验证A、B两种教学方法对学生成绩的差异，需用独立样本t检验并绘制对比柱状图"
        )
        
        if st.button("提交分析请求", type="secondary", use_container_width=True) and user_query:
            with st.spinner("🤖 正在理解你的科研意图并执行分析..."):
                analysis_agent = get_analysis_agent(df)
                result = analysis_agent.invoke({"input": user_query})
                
                st.markdown("### 💡 科研分析结果")
                st.markdown(result["output"], unsafe_allow_html=True)
                
                if os.path.exists("plot.png"):
                    st.markdown("### 📈 结果可视化")
                    st.image("plot.png", use_container_width=True)
                    os.remove("plot.png")
    
    # 报告下载功能
    if "result" in locals() and result.get("output"):
        st.divider()
        report_content = f"""# AI科研数据分析报告
## 报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 数据概况：{len(df)}行 × {len(df.columns)}列
## 分析需求：{auto_query if 'auto_query' in locals() else user_query}
## 完整分析结果：
{result['output']}
"""
        st.download_button(
            label="📥 下载分析报告（Markdown格式）",
            data=report_content,
            file_name=f"AI科研分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.md",
            mime="text/markdown"
        )

# 无数据时的提示
else:
    st.info("💡 请在左侧边栏上传CSV/Excel文件（支持多文件关联），上传后自动加载数据概况")
