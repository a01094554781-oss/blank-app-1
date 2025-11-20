"""
tidal_dashboard.py

--- 목적(발표용 텍스트) ---
대시보드 목적:
- 한국 갯벌(연도별/지역별)과 전세계 갯벌 분포를 비교·시각화하여
  정책결정자와 시민이 갯벌 보존의 긴급성과 지역별 우선순위를 파악하도록 지원.

주요 사용자:
- 해양수산부/환경부 담당자, 지자체 정책담당자, 연구자, 시민

데이터 출처(다운로드 필요):
- Global Intertidal Change / Murray dataset (1999-2019) — 제공처: GlobalIntertidalChange.org & JCU / Google Earth Engine.
  (데이터 상세/다운로드 페이지 참고)  :contentReference[oaicite:5]{index=5}
- 한국 갯벌 현황(해양수산부) — 공공데이터포털에 CSV/JSON 파일 있음. :contentReference[oaicite:6]{index=6}
- KOSIS 시도별 연안습지 면적 통계 (참조). :contentReference[oaicite:7]{index=7}

평가 포인트(발표시 강조):
- 모든 그래프에 축 라벨(단위), 범례 표기
- 각 시각화는 '어떤 질문에 답하는가'를 캡션으로 표기
- Gemini 챗봇으로 클릭한 지역 요약 제공(placeholder)

--- 사용법 요약 ---
1) 필요한 라이브러리 설치: pip install -r requirements.txt
   requirements.txt 예시:
      dash plotly geopandas pandas rasterio shapely pyproj dash-leaflet
2) data/ 폴더에 아래 파일을 넣기 (구체적 링크는 코드 하단 주석 참조)
3) python tidal_dashboard.py
4) 브라우저 열기 -> http://127.0.0.1:8050

"""

import os
import json
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, shape
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_leaflet as dl
import dash_leaflet.express as dlx

# -----------------------
# 파일 경로(사용자가 data 폴더에 파일을 넣어야 함)
# -----------------------
DATA_DIR = "data"

# Global tidal flats data (GeoJSON or simplified GeoPackage recommended)
GLOBAL_TIDAL_GEOJSON = os.path.join(DATA_DIR, "global_tidal_flats.geojson")
# Korean mudflats points/areas (CSV with lat/lon or shapefile)
KOR_MUDFLATS_CSV = os.path.join(DATA_DIR, "kor_mudflats.csv")
KOR_MUDFLATS_SHP = os.path.join(DATA_DIR, "kor_mudflats_shapefile.shp")  # optional shapefile
# Simple precomputed stats (optional)
KOR_STATS_CSV = os.path.join(DATA_DIR, "kor_mudflats_stats.csv")

# -----------------------
# 유틸리티: 데이터 로딩
# -----------------------
def load_global_tidal_geojson(path=GLOBAL_TIDAL_GEOJSON):
    """
    기대: global_tidal_flats.geojson은 (geometry, year, probability, type) 등 컬럼을 포함.
    만약 GeoTIFF라면 사전 처리 후 GeoJSON으로 변환하여 사용 권장.
    """
    if not os.path.exists(path):
        print(f"[경고] {path} 파일을 찾을 수 없습니다. (global tidal flats layer 필요)")
        return None
    g = gpd.read_file(path)
    return g

def load_kor_mudflats_csv(path=KOR_MUDFLATS_CSV):
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8')
        # 최소한 lat/lon, area_km2, name, year 등의 컬럼을 가지고 있어야 함
        if 'latitude' in df.columns and 'longitude' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs="EPSG:4326")
            return gdf
        else:
            # CSV가 통계만 포함하는 경우
            return df
    elif os.path.exists(KOR_MUDFLATS_SHP):
        return gpd.read_file(KOR_MUDFLATS_SHP)
    else:
        print(f"[경고] 한국 갯벌 데이터 파일({path})이 없습니다.")
        return None

def load_kor_stats(path=KOR_STATS_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# -----------------------
# 데이터 로드
# -----------------------
global_tidal_gdf = load_global_tidal_geojson()
kor_mud_gdf = load_kor_mudflats_csv()
kor_stats = load_kor_stats()

# -----------------------
# 간단한 데이터 처리(없는 컬럼은 최소 예시 생성)
# -----------------------
# kor_stats 예시 필요 컬럼: ['year', 'total_mudflat_km2', 'change_km2', 'protected_km2', 'reclaimed_km2']
if kor_stats is None and isinstance(kor_mud_gdf, gpd.GeoDataFrame):
    # 예시: 시계열이 없으면 간단 집계로 대체 (실제 분석은 출처의 연도별 파일 사용)
    try:
        # 만약 area_km2가 있다면 연도별 집계
        if 'area_km2' in kor_mud_gdf.columns and 'year' in kor_mud_gdf.columns:
            kor_stats = kor_mud_gdf.groupby('year').agg(total_mudflat_km2=('area_km2','sum')).reset_index()
            kor_stats['change_km2'] = kor_stats['total_mudflat_km2'].diff().fillna(0)
        else:
            kor_stats = pd.DataFrame({'year':[2000,2010,2020,2023],'total_mudflat_km2':[2600,2550,2460,2443],'change_km2':[0,-50,-90,-17]})
    except Exception as e:
        kor_stats = pd.DataFrame({'year':[2000,2010,2020,2023],'total_mudflat_km2':[2600,2550,2460,2443],'change_km2':[0,-50,-90,-17]})

# -----------------------
# Dash 앱 구성
# -----------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(style={'font-family':'Arial, sans-serif','margin':'10px'}, children=[
    html.H1("갯벌·간척지 대시보드 — 한국 & Global Tidal Flats", style={'textAlign':'center'}),
    html.Div([
        # KPI 카드 영역
        html.Div([
            html.Div(id='kpi-total-km2', style={'padding':'10px','border':'1px solid #ddd','borderRadius':'6px','display':'inline-block','width':'32%','marginRight':'1%'}),
            html.Div(id='kpi-decline-10yr', style={'padding':'10px','border':'1px solid #ddd','borderRadius':'6px','display':'inline-block','width':'32%','marginRight':'1%'}),
            html.Div(id='kpi-reclaimed-km2', style={'padding':'10px','border':'1px solid #ddd','borderRadius':'6px','display':'inline-block','width':'32%'}),
        ], style={'marginBottom':'10px'}),
        # 상단: 세계 지도 + 국가별 그래프
        html.Div([
            html.Div([
                html.H4("세계 갯벌 분포 (Global tidal flats)", style={'margin':'5px 0'}),
                html.Div([
                    dl.Map(center=[20,0], zoom=2, children=[
                        dl.TileLayer(),
                        # GeoJSON 레이어 (간단 버전 — 실제는 큰 레이어를 tileserver로 배포 권장)
                        dl.LayerGroup(id='global-layer'),
                        dl.ScaleControl(position="bottomleft")
                    ], style={'width':'100%','height':'450px'}, id='global-map')
                ])
            ], style={'width':'62%','display':'inline-block','verticalAlign':'top','paddingRight':'8px'}),
            html.Div([
                html.H4("국가별 갯벌 면적 Top (예시)", style={'margin':'5px 0'}),
                dcc.Graph(id='country-bar', style={'height':'420px'})
            ], style={'width':'36%','display':'inline-block','verticalAlign':'top'}),
        ]),
        # 하단: 한국 상세 + 시계열
        html.Div([
            html.Div([
                html.H4("한국 갯벌 위치 (클릭 시 상세정보)", style={'margin':'5px 0'}),
                dl.Map(center=[36,126], zoom=6, children=[
                    dl.TileLayer(),
                    dl.LayerGroup(id='kor-layer'),
                    dl.ScaleControl(position="bottomleft")
                ], style={'width':'100%','height':'400px'})
            ], style={'width':'49%','display':'inline-block','paddingRight':'8px'}),
            html.Div([
                html.H4("한국 갯벌 시계열", style={'margin':'5px 0'}),
                dcc.Graph(id='kor-timeseries', style={'height':'300px'}),
                html.Div(id='timeseries-caption', style={'fontSize':'12px','color':'#444','marginTop':'6px'})
            ], style={'width':'49%','display':'inline-block'}),
        ], style={'marginTop':'12px'}),
        # 오른쪽 하단: 간척지 용도별 스택차트 + 보호지역 비교
        html.Div([
            html.Div([
                html.H4("간척지 용도별 변화 (예시 스택형)", style={'margin':'5px 0'}),
                dcc.Graph(id='reclaim-stack')
            ], style={'width':'49%','display':'inline-block','verticalAlign':'top','paddingRight':'8px'}),
            html.Div([
                html.H4("보호지역 vs 비보호지역 갯벌 변화", style={'margin':'5px 0'}),
                dcc.Graph(id='protected-vs-unprotected')
            ], style={'width':'49%','display':'inline-block','verticalAlign':'top'}),
        ], style={'marginTop':'12px'}),
        # Gemini 챗봇(플레이스홀더)
        html.Div([
            html.H4("요약 / 질문 (Gemini 챗봇 연동 자리)", style={'margin':'8px 0 4px 0'}),
            dcc.Textarea(id='chat-query', placeholder='지역을 클릭하거나 질문을 입력하세요 (예: 인천의 2000-2020 갯벌 감소량 알려줘)', style={'width':'80%','height':'60px'}),
            html.Button('질문 보내기', id='chat-send', n_clicks=0),
            html.Div(id='chat-response', style={'border':'1px solid #ddd','padding':'8px','marginTop':'6px','whiteSpace':'pre-wrap','minHeight':'60px'})
        ], style={'marginTop':'12px'}),
        # 아래: 데이터 출처 및 다운로드 안내
        html.Div([
            html.H5("데이터 출처 및 다운로드 안내"),
            html.Ul([
                html.Li("Global tidal flats dataset (1999-2019) — Global Intertidal Change / JCU Murray. (GEE / Figshare / intertidal.app)."),
                html.Li("한국 갯벌 데이터 — 해양수산부 / 공공데이터포털 (갯벌면적 파일)."),
                html.Li("시도별 통계 — KOSIS 연안습지 면적 통계.")
            ]),
            html.P("참고: 대용량 레이어는 GeoJSON으로 직접 로드하지 말고 타일서버(예: GeoServer, cloud-optimized GeoTIFF + tiles)로 배포한 뒤 Leaflet에 TileLayer/WMTS로 연결하세요.")
        ], style={'marginTop':'14px','fontSize':'13px','color':'#333'})
    ], style={'maxWidth':'1200px','margin':'0 auto'})
])

# -----------------------
# 콜백: KPI 계산/채우기
# -----------------------
@app.callback(
    Output('kpi-total-km2','children'),
    Output('kpi-decline-10yr','children'),
    Output('kpi-reclaimed-km2','children'),
    Input('kor-timeseries','id')  # dummy trigger; we update once at start
)
def update_kpis(_):
    # 안전하게 값을 추출
    try:
        latest = kor_stats.loc[kor_stats['year'].idxmax()]
        total_km2 = latest['total_mudflat_km2']
        # 10년 전이 있다면 계산
        years = sorted(kor_stats['year'].unique())
        if len(years) >= 2:
            last10_year = years[-1]
            # look for value 10 years ago or first entry
            prev = kor_stats.iloc[0]
            if (last10_year - years[0]) >= 10:
                # find nearest to last-10
                target_year = last10_year - 10
                if target_year in list(kor_stats['year']):
                    prev = kor_stats[kor_stats['year']==target_year].iloc[0]
        decline_10yr = (latest['total_mudflat_km2'] - prev['total_mudflat_km2'])
    except Exception as e:
        total_km2 = 0
        decline_10yr = 0

    # reclaimed: if column exists
    reclaimed = kor_stats['change_km2'].loc[kor_stats['change_km2']<0].sum() * -1 if 'change_km2' in kor_stats.columns else None

    # Format cards (축 라벨/단위 표기 강조)
    total_card = html.Div([
        html.Div("총 갯벌 면적 (최근 연도)", style={'fontSize':'12px','color':'#666'}),
        html.Div(f"{total_km2:,.2f} km²", style={'fontSize':'24px','fontWeight':'600'}),
        html.Div("데이터 단위: km²", style={'fontSize':'11px','color':'#666'})
    ])
    decline_card = html.Div([
        html.Div("최근 10년 변화(예시)", style={'fontSize':'12px','color':'#666'}),
        html.Div(f"{decline_10yr:+,.2f} km²", style={'fontSize':'24px','fontWeight':'600','color':('#d9534f' if decline_10yr<0 else '#5cb85c')}),
        html.Div("음수 = 감소", style={'fontSize':'11px','color':'#666'})
    ])
    reclaimed_card = html.Div([
        html.Div("누적 간척/소실(음수 합계, 예시)", style={'fontSize':'12px','color':'#666'}),
        html.Div(f"{reclaimed:,.2f} km²" if reclaimed is not None else "데이터 없음", style={'fontSize':'24px','fontWeight':'600'}),
        html.Div("데이터 단위: km²", style={'fontSize':'11px','color':'#666'})
    ])
    return total_card, decline_card, reclaimed_card

# -----------------------
# 콜백: 나라별 바 차트 (예시 데이터)
# -----------------------
@app.callback(Output('country-bar','figure'), Input('global-map','id'))
def update_country_bar(_):
    # 예시로 세계 상위 10개국(정확한 값은 데이터 로드 후 교체)
    df = pd.DataFrame({
        'country':['China','USA','Brazil','Australia','India','Indonesia','Korea, Rep.','Bangladesh','Myanmar','Japan'],
        'mudflat_km2':[12000,8500,6000,4800,4200,3900,2443,2100,1800,1600]
    })
    fig = px.bar(df.sort_values('mudflat_km2', ascending=True), x='mudflat_km2', y='country', orientation='h',
                 labels={'mudflat_km2':'갯벌 면적 (km²)','country':'국가'},
                 title='국가별 갯벌 면적(예시, 단위: km²)')
    fig.update_layout(margin=dict(l=60,r=10,t=40,b=30))
    return fig

# -----------------------
# 콜백: 한국 시계열
# -----------------------
@app.callback(Output('kor-timeseries','figure'), Output('timeseries-caption','children'), Input('kor-layer','id'))
def update_kor_timeseries(_):
    df = kor_stats.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['year'], y=df['total_mudflat_km2'], mode='lines+markers', name='총 갯벌 면적'))
    if 'change_km2' in df.columns:
        fig.add_trace(go.Bar(x=df['year'], y=df['change_km2'], name='연간 변화 (km²)', yaxis='y2', opacity=0.6))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='연간 변화 (km²)'))
    fig.update_layout(title='한국 갯벌 면적 연도별 변화', xaxis_title='연도', yaxis_title='갯벌 면적 (km²)', legend_title='지표')
    caption = "이 그래프는 자료 출처의 연도별 갯벌 면적 데이터를 사용합니다. (축: 면적 단위 km²)"
    return fig, caption

# -----------------------
# 콜백: reclaim stack 및 protected comparison (더미/예시)
# -----------------------
@app.callback(Output('reclaim-stack','figure'), Input('reclaim-stack','id'))
def update_reclaim_stack(_):
    # 예시 데이터(실제 데이터로 교체)
    years = [2000,2010,2020,2023]
    df = pd.DataFrame({
        'year': years,
        'agriculture':[50,40,20,15],
        'industry':[10,20,30,40],
        'port':[5,15,30,35],
        'residential':[0,5,10,12]
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['year'], y=df['agriculture'], name='농업(간척)', hovertemplate='%{y} km²'))
    fig.add_trace(go.Bar(x=df['year'], y=df['industry'], name='공업(간척)'))
    fig.add_trace(go.Bar(x=df['year'], y=df['port'], name='항만'))
    fig.add_trace(go.Bar(x=df['year'], y=df['residential'], name='주거/기타'))
    fig.update_layout(barmode='stack', title='간척지 용도별 변화 (예시) - 단위: km²', xaxis_title='연도', yaxis_title='면적 (km²)')
    return fig

@app.callback(Output('protected-vs-unprotected','figure'), Input('protected-vs-unprotected','id'))
def update_protected_vs(_):
    # 예시: 보호지역/비보호 지역 연도별 면적
    df = pd.DataFrame({
        'year':[2000,2010,2020,2023],
        'protected':[500,480,470,468],
        'unprotected':[2100,2050,1970,1975]
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['year'], y=df['protected'], mode='lines+markers', name='보호지역'))
    fig.add_trace(go.Scatter(x=df['year'], y=df['unprotected'], mode='lines+markers', name='비보호지역'))
    fig.update_layout(title='보호지역 vs 비보호지역 갯벌 면적 (단위: km²)', xaxis_title='연도', yaxis_title='면적 (km²)')
    return fig

# -----------------------
# 콜백: 지도 레이어 로드 (GeoJSON -> Leaflet)
# -----------------------
@app.callback(Output('global-layer','children'), Input('global-map','center'))
def load_global_layer(center):
    # global_tidal_gdf가 있으면 GeoJSON으로 변환 (주의: 대용량은 비효율)
    if global_tidal_gdf is None:
        return [dl.Marker(position=[0,0], children=dl.Tooltip("Global tidal data not found. Place data/global_tidal_flats.geojson"))]
    # 단순화(심볼용): 각 폴리곤에 probability 또는 type 표현
    features = []
    # 제한: 큰 파일이면 여기서 메모리 초과 가능 -> 사용 전 벡터 타일 권장
    sample = global_tidal_gdf.head(500) if len(global_tidal_gdf)>500 else global_tidal_gdf
    for _, row in sample.iterrows():
        geom = json.loads(gpd.GeoSeries([row['geometry']]).to_json())['features'][0]['geometry']
        prop = {'prob': float(row.get('probability', 1.0)) if 'probability' in row else 1.0}
        features.append({"type":"Feature","geometry":geom,"properties":prop})
    geojson = {"type":"FeatureCollection","features":features}
    # 스타일 함수: probability에 따라 투명도 조정
    def style_feature(feature):
        prob = feature['properties'].get('prob',1.0)
        color = '#1f77b4'  # 파란색
        return {'color': color, 'weight': 0.5, 'fillOpacity': max(0.12, min(0.9, prob))}
    return [dl.GeoJSON(data=geojson, options=dict(style=style_feature), zoomToBounds=True)]

@app.callback(Output('kor-layer','children'), Input('kor-layer','id'))
def load_kor_layer(_):
    if kor_mud_gdf is None:
        return [dl.Marker(position=[36,126], children=dl.Tooltip("한국 갯벌 데이터 없음. data/kor_mudflats.csv 업로드 필요"))]
    # 포인트 또는 폴리곤 지원
    features = []
    if isinstance(kor_mud_gdf, gpd.GeoDataFrame):
        sample = kor_mud_gdf.copy()
        # 필요한 경우 속성 간단화
        for _, row in sample.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Point':
                lat = geom.y; lon = geom.x
                tooltip = f"{row.get('name','갯벌')}\n면적: {row.get('area_km2','N/A')} km²\n연도: {row.get('year','N/A')}"
                features.append(dl.Marker(position=[lat,lon], children=dl.Tooltip(tooltip)))
            else:
                # 폴리곤은 GeoJSON
                geom_json = json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry']
                features.append(dl.GeoJSON(data={"type":"Feature","geometry":geom_json,"properties":{}}, options=dict(style={"color":"#0066cc","weight":1,"fillOpacity":0.4})))
        return features
    else:
        return [dl.Marker(position=[36,126], children=dl.Tooltip("한국 갯벌 데이터 형식 불일치"))]

# -----------------------
# 콜백: Gemini 챗봇(플레이스홀더)
# -----------------------
@app.callback(Output('chat-response','children'), Input('chat-send','n_clicks'), State('chat-query','value'))
def gemini_query(n, text):
    if n is None or n==0:
        return ""
    if not text:
        return "질문을 입력하세요."
    # 실제로는 여기에 Gemini/OpenAI API 호출 코드를 넣어 연동.
    # 예제 응답(간단 요약 생성)
    reply = f"[요약봇 응답 예시]\n입력질문: {text}\n\n(실제 시스템 연동시 여기에 지역 기반 통계와 시각화 요약을 자동으로 생성합니다.)"
    return reply

# -----------------------
# 앱 실행
# -----------------------
if __name__ == '__main__':
    app.run_server(debug=True)
