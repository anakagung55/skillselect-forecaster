import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet # Pastikan library ini ada

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Interlace Gacor | AI Forecast", 
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stMetric {
        background-color: #0E1117;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
    }
    .stMultiSelect, .stSelectbox {
        margin-bottom: 15px;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA (CACHE)
# ==========================================
@st.cache_data
def get_master_data():
    # Prioritaskan Parquet, fallback ke CSV/Zip
    if os.path.exists('df_master.parquet'):
        df = pd.read_parquet('df_master.parquet')
    elif os.path.exists('df_master.zip'):
        df = pd.read_csv('df_master.zip')
    elif os.path.exists('data/df_master.csv'):
        df = pd.read_csv('data/df_master.csv')
    else:
        return pd.DataFrame() # Return kosong jika file tidak ada

    # Data Cleaning Ringan
    df['ds'] = pd.to_datetime(df['ds'])
    df['visa_type'] = df['visa_type'].astype(str)
    df['eoi_status'] = df['eoi_status'].astype(str)
    df['points'] = df['points'].astype(str)
    return df

df = get_master_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🎮 Control Panel")

page = st.sidebar.radio("Navigation", [
    "Project Overview", 
    "🏆 Top Market Leaderboard", 
    "🔮 Specific Forecast & Trends"
], key='page')

# Callbacks for simple button navigation (use on_click to avoid session_state write errors)
def _go_to_leaderboard():
    st.session_state['page'] = '🏆 Top Market Leaderboard'

def _go_to_forecast():
    st.session_state['page'] = '🔮 Specific Forecast & Trends'

# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================
if page == "Project Overview":
    st.title("🚀 SkillSelect — Visa Application Trends Dashboard (EOI)")
    st.subheader("Interactive dashboard to explore trends and simple forecasts")
    
    st.markdown("""
    This app helps non-technical users understand visa application trends (EOI).
    - See applicant counts by occupation, compare time periods, and spot rising or falling trends.
    - Get a simple 6-month forecast to help with planning.
    Just select an occupation and filters, then view clear charts and summaries.
    """)

    st.write("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        unique_occ = len(df['occupation'].unique()) if not df.empty else 0
        st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>{unique_occ}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><b>Number of occupations</b></p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<h3 style='text-align: center; color: #2196F3;'>Jan 24 - Dec 25</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><b>Data period</b></p>", unsafe_allow_html=True)
    with col3:
        total_eoi = int(df['count_eois'].sum()) if not df.empty else 0
        st.markdown(f"<h3 style='text-align: center; color: #FF9800;'>{total_eoi:,}</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><b>Total applications (EOI)</b></p>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<h3 style='text-align: center; color: #9C27B0;'>95%</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'><b>Model confidence</b></p>", unsafe_allow_html=True)
    
    st.write("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📊 Global EOI Trend (2024 - 2025)")
        if not df.empty:
            trend_data = df.groupby('ds')['count_eois'].sum().reset_index()
            st.line_chart(trend_data.set_index('ds'))
            st.caption("This chart shows total EOI applications per period — useful to spot rising and falling trends.")

    with col_right:
        st.subheader("🎯 App purpose")
        with st.expander("Why this is useful", expanded=True):
            st.write("""
            1. Collect SkillSelect data into one place for easy access.
            2. Show seasonal patterns and trends so you can see when applications rise or fall.
            3. Provide forecasts that help with planning (e.g., staffing or process load).
            """)
        
        st.subheader("📂 Data source")
        st.info("Data comes from **SkillSelect - Australian Government** and covers subclasses **189, 190, and 491**.")

    st.subheader("🔄 How it works (brief)")
    step1, step2 = st.columns(2)
    step1.success("**1. Automatic model training**\n\nA model is trained from historical data to produce simple forecasts.")
    step2.success("**2. Easy visualizations**\n\nInteractive charts help you understand trends without manual data work.")

    st.write("---")
    st.subheader("💡 Quick Insights & Actions")
    if not df.empty:
        # 1) Top 5 in the latest period
        latest_date = df['ds'].max()
        top_latest = df[df['ds'] == latest_date].groupby('occupation')['count_eois'].sum().nlargest(5)
        st.markdown(f"**Top 5 occupations in the latest period ({latest_date.strftime('%B %Y')}):**")
        st.table(top_latest.reset_index().rename(columns={'count_eois':'Applicants'}))

        # 2) Pertumbuhan: rata-rata 3 bulan terakhir vs 3 bulan sebelumnya
        df_monthly = df.groupby([pd.Grouper(key='ds', freq='MS'), 'occupation'])['count_eois'].sum().reset_index()
        months = sorted(df_monthly['ds'].dt.to_period('M').unique()) if not df_monthly.empty else []
        if len(months) >= 6:
            last3 = months[-3:]
            prev3 = months[-6:-3]
            last_avg = df_monthly[df_monthly['ds'].dt.to_period('M').isin(last3)].groupby('occupation')['count_eois'].mean()
            prev_avg = df_monthly[df_monthly['ds'].dt.to_period('M').isin(prev3)].groupby('occupation')['count_eois'].mean()
            growth = ((last_avg - prev_avg) / prev_avg.replace({0: pd.NA})) * 100
            growth = growth.dropna().sort_values(ascending=False).head(5).reset_index()
            growth.columns = ['Occupation', 'Growth (%)']
            growth['Growth (%)'] = growth['Growth (%)'].map(lambda x: f"{x:.1f}%")
            st.markdown("**Occupations with highest growth (avg last 3 months vs previous 3 months):**")
            st.table(growth)
        else:
            st.info("At least 6 months of monthly data required to compute a reliable growth trend.")

        st.write("---")
        jump_col1, jump_col2 = st.columns([2,2])
        with jump_col1:
            st.button("➡️ View Top Occupations", on_click=_go_to_leaderboard)
        with jump_col2:
            st.button("➡️ Go to Forecast Page", on_click=_go_to_forecast)
    else:
        st.info("No data available. Add `df_master.parquet` or a CSV in `data/` to see insights.")


# ==========================================
# PAGE 2: TOP MARKET LEADERBOARD
# ==========================================
elif page == "🏆 Top Market Leaderboard":
    st.title("🏆 Top Occupations Ranking")
    st.markdown("Shows occupations with the highest number of applicants. Use filters to view by visa type, status, or time period.")
    
    if not df.empty:
        df['month_year'] = df['ds'].dt.strftime('%Y-%m') 
        all_months = sorted(df['month_year'].unique())
        
        with st.expander("🎛️ Filter Settings (click to open)", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                all_visas = sorted(df['visa_type'].unique())
                sel_visas = st.multiselect("Visa Type (select one or more):", all_visas, default=all_visas)
            with f_col2:
                all_status = sorted(df['eoi_status'].unique())
                def_status = ['SUBMITTED'] if 'SUBMITTED' in all_status else all_status
                sel_status = st.multiselect("EOI Status (e.g., SUBMITTED):", all_status, default=def_status)
            with f_col3:
                sel_months = st.multiselect("Period (YYYY-MM) (pick months):", all_months, default=all_months)
        
        df_filtered = df[
            (df['visa_type'].isin(sel_visas)) &
            (df['eoi_status'].isin(sel_status)) &
            (df['month_year'].isin(sel_months))
        ].copy()

        df_filtered['points_num'] = pd.to_numeric(df_filtered['points'], errors='coerce')

        if df_filtered.empty:
            st.warning("No data for this filter combination. Try different filters.")
        else:
            df_ranking = df_filtered.groupby('occupation')['count_eois'].sum().reset_index()
            df_top15 = df_ranking.sort_values(by='count_eois', ascending=False).head(15)
            
            leaderboard_data = []
            for occ in df_top15['occupation']:
                occ_data = df_filtered[df_filtered['occupation'] == occ]
                total_vol = occ_data['count_eois'].sum()
                
                if not occ_data.empty and occ_data['points_num'].notna().any():
                    pt_dist = occ_data.groupby('points_num')['count_eois'].sum()
                    dom_point = pt_dist.idxmax() if not pt_dist.empty else 0
                    min_point = occ_data['points_num'].min()
                    max_point = occ_data['points_num'].max()
                else:
                    dom_point, min_point, max_point = 0, 0, 0
                    
                leaderboard_data.append({
                    "Occupation": occ,
                    "Total Demand": int(total_vol),
                    "Dominant Point": int(dom_point),
                    "Lowest Point": int(min_point),
                    "Highest Point": int(max_point)
                })
                
            df_display = pd.DataFrame(leaderboard_data)

            st.divider()
            st.subheader("📊 Top 15 Occupations (Based on filter)")
            fig_bar = px.bar(
                df_display, x='Total Demand', y='Occupation', orientation='h',
                text='Total Demand', color='Total Demand',
                color_continuous_scale='Viridis', template='plotly_dark', height=600
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("📋 Points summary")
            st.dataframe(
                df_display.set_index('Occupation'),
                column_config={
                    "Total Demand": st.column_config.NumberColumn("Total Applicants", format="%d 👤"),
                    "Dominant Point": st.column_config.ProgressColumn("Dominant Score", format="%d", min_value=0, max_value=130),
                    "Lowest Point": st.column_config.NumberColumn("Min Score", format="%d"),
                    "Highest Point": st.column_config.NumberColumn("Max Score", format="%d")
                },
                use_container_width=True
            )
    else:
        st.error("Data frame is empty. Make sure the parquet file is readable.")


# ==========================================
# PAGE 3: SPECIFIC FORECAST (REAL-TIME TRAINING)
# ==========================================
elif page == "🔮 Specific Forecast & Trends":
    st.title("🔮 Forecast for Selected Occupation")
    st.markdown("Select an occupation to view historical data, breakdown by visa type, and a 6-month forecast. No technical knowledge required — just choose and click Forecast.")

    if df.empty:
        st.error("Data Master tidak termuat.")
    else:
        # 1. PILIH PEKERJAAN DARI DATA (BUKAN DARI FOLDER MODEL)
        available_occ = sorted(df['occupation'].unique())
        selected_occ = st.selectbox("Select the occupation you want to view (e.g., Software Engineer):", available_occ, key='selected_occ')
        
        # Filter Data Khusus Pekerjaan Terpilih
        df_occ = df[df['occupation'] == selected_occ].copy()
        
        st.divider()

        # 2. FILTER & POINTS ANALYSIS
        st.subheader("2️⃣ Historical Data Analysis — breakdown by visa type")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            all_visas = sorted(df_occ['visa_type'].unique())
            sel_visas = st.multiselect("Visa Type (select):", all_visas, default=all_visas, key="fore_visa")
        with col_f2:
            all_status = sorted(df_occ['eoi_status'].unique())
            def_status = ['SUBMITTED'] if 'SUBMITTED' in all_status else all_status
            sel_status = st.multiselect("EOI Status (select):", all_status, default=def_status, key="fore_status")
        with col_f3:
            all_points = sorted(df_occ['points'].unique())
            sel_points = st.multiselect("Points range (select):", all_points, default=all_points, key="fore_points")

        df_filtered = df_occ[
            (df_occ['visa_type'].isin(sel_visas)) &
            (df_occ['eoi_status'].isin(sel_status)) &
            (df_occ['points'].isin(sel_points))
        ]

        df_chart = df_filtered.groupby(['ds', 'visa_type'])['count_eois'].sum().reset_index()
        total_filtered = df_filtered.groupby('ds')['count_eois'].sum().iloc[-1] if not df_chart.empty else 0

        c_chart, c_metric = st.columns([3, 1])
        with c_chart:
            if df_chart.empty:
                st.warning("No data for the selected filters. Try other options.")
            else:
                fig_breakdown = px.bar(
                    df_chart, x='ds', y='count_eois', color='visa_type', 
                    title=f"Komposisi EOI: {selected_occ}",
                    template='plotly_dark', height=400
                )
                st.plotly_chart(fig_breakdown, use_container_width=True)
        
        with c_metric:
            st.metric("Current total (filtered)", f"{int(total_filtered):,}")
            st.info("The chart shows real data matching the filters you selected.")

        st.divider()

        # 3. AI FORECAST (REAL-TIME TRAINING)
        st.subheader("3️⃣ Future projection (6 months)")
        
        if st.button("Generate AI forecast (6 months) 🚀", type="primary", use_container_width=True):
            with st.spinner('Training model and computing future projections...'):
                
                # --- DATA PREPARATION ---
                # Use total occupation data (no visa/status filter for global prediction)
                df_train = df_occ.groupby('ds')['count_eois'].sum().reset_index()
                df_train.columns = ['ds', 'y']
                
                if len(df_train) < 2:
                    st.error("Not enough historical data (< 2 months) to run an AI forecast.")
                else:
                    # --- TRAINING PROPHET (ON-THE-FLY) ---
                    try:
                        model = Prophet(seasonality_mode='multiplicative')
                        model.fit(df_train)
                        
                        future = model.make_future_dataframe(periods=6, freq='MS') # Prediksi 6 bulan
                        forecast = model.predict(future)
                        
                        # --- VISUALISASI ---
                        fig_ai = go.Figure()
                        
                        # Trace A: Actual Data
                        fig_ai.add_trace(go.Scatter(
                            x=df_train['ds'], 
                            y=df_train['y'],
                            mode='markers+lines',
                            name='Actual data (history)',
                            line=dict(color='#00CC96', width=2),
                            marker=dict(size=6)
                        ))

                        # Trace B: AI Forecast
                        fig_ai.add_trace(go.Scatter(
                            x=forecast['ds'], 
                            y=forecast['yhat'], 
                            mode='lines', 
                            name='AI forecast (future)', 
                            line=dict(color='#636EFA', width=3)
                        ))

                        # Trace C: Confidence Interval
                        fig_ai.add_trace(go.Scatter(
                            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]), 
                            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]), 
                            fill='toself', 
                            fillcolor='rgba(99, 110, 250, 0.2)', 
                            line=dict(color='rgba(0,0,0,0)'), 
                            name='Confidence interval'
                        ))
                        
                        fig_ai.update_layout(
                            title=f"Forecast: {selected_occ}", 
                            template="plotly_dark", 
                            height=450, 
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig_ai, use_container_width=True)
                        
                        # Show table
                        res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6)
                        res.columns = ['Month', 'Forecast', 'Lower bound', 'Upper bound']
                        res['Month'] = res['Month'].dt.strftime('%B %Y')
                        st.table(res)
                        
                    except Exception as e:
                        st.error(f"Failed to train model: {e}")
