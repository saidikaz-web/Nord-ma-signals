import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

st.set_page_config(page_title='NORD • MA Signals', page_icon='📈', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif} .stApp{background:#0a101b;color:#e7edf7}
.block-container{max-width:1250px;padding-top:1.4rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em}
h1{font-size:clamp(2rem,5vw,3.3rem)!important}
[data-testid="stMetric"]{background:#141e2d;border:1px solid #27374c;padding:17px;border-radius:15px}
[data-testid="stMetricLabel"]{color:#9cacbf} [data-testid="stMetricValue"]{color:#e7f0ff}
div.stButton>button[kind="primary"]{background:#66e1b6;color:#081611;border:0;font-weight:700}
div.stButton>button{border-radius:10px;min-height:44px}
[data-testid="stDataFrame"]{border:1px solid #28384b;border-radius:12px;overflow:hidden}
.hero{padding:20px 24px;border:1px solid #293d51;border-radius:18px;background:linear-gradient(120deg,#14293a,#101825 68%);margin-bottom:18px}
.kicker{color:#6ee2b8;font-size:.76rem;letter-spacing:.19em;font-weight:800}
.muted{color:#9eafc2}.pill{display:inline-block;border:1px solid #3d655e;color:#9df5d1;padding:5px 10px;border-radius:99px;font-size:.8rem}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:16px}h1{font-size:2rem!important}}
</style>''', unsafe_allow_html=True)

TICKERS = {'Sandvik':'SAND.ST','Volvo B':'VOLV-B.ST','Atlas Copco A':'ATCO-A.ST','Atlas Copco B':'ATCO-B.ST','Epiroc B':'EPI-B.ST','Saab B':'SAAB-B.ST','Boliden':'BOL.ST','Investor B':'INVE-B.ST','Assa Abloy B':'ASSA-B.ST','Alfa Laval':'ALFA.ST','SKF B':'SKF-B.ST','SEB A':'SEB-A.ST','Swedbank A':'SWED-A.ST','Ericsson B':'ERIC-B.ST','H&M B':'HM-B.ST','Hexagon B':'HEXA-B.ST','Nibe B':'NIBE-B.ST','Sinch':'SINCH.ST','SSAB A':'SSAB-A.ST','Getinge B':'GETI-B.ST','Lifco B':'LIFCO-B.ST','Indutrade':'INDT.ST','AstraZeneca':'AZN.ST','Tele2 B':'TEL2-B.ST','Electrolux B':'ELUX-B.ST','Evolution':'EVO.ST','Addtech B':'ADDT-B.ST','Avanza':'AZA.ST','SCA B':'SCA-B.ST','Latour B':'LATO-B.ST','Beijer Ref B':'BEIJ-B.ST','Truecaller B':'TRUE-B.ST'}

@st.cache_data(ttl=1800,show_spinner=False)
def load(ticker):
    d=yf.download(ticker,period='2y',interval='1d',auto_adjust=True,progress=False,threads=False)
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    if d.empty or len(d)<220:raise ValueError('Otillräcklig historik')
    d=d[['Open','High','Low','Close','Volume']].dropna().copy()
    c,h,l,v=d.Close,d.High,d.Low,d.Volume
    for n in [20,50,200]:d[f'MA{n}']=c.rolling(n).mean()
    prev=c.shift(1);tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    up=h.diff();down=-l.diff()
    pdm=up.where((up>down)&(up>0),0.);mdm=down.where((down>up)&(down>0),0.)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    pdi=100*pdm.ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
    mdi=100*mdm.ewm(alpha=1/14,adjust=False).mean()/atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d['ADX']=dx.ewm(alpha=1/14,adjust=False).mean()
    delta=c.diff();gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d['RSI']=100-100/(1+gain/loss.replace(0,np.nan))
    d['ATR%']=100*atr/c
    d['ADR%']=(100*(h-l)/l.replace(0,np.nan)).rolling(14).mean()
    d['RelVol']=v/v.shift(1).rolling(20).mean()
    d['Oms MSEK']=(c*v).rolling(20).mean()/1e6
    d['CrossUp']=(d.MA20>d.MA50)&(d.MA20.shift(1)<=d.MA50.shift(1))
    d['CrossDown']=(d.MA20<d.MA50)&(d.MA20.shift(1)>=d.MA50.shift(1))
    return d

st.markdown('<div class="hero"><div class="kicker">NORD / TECHNICAL SCANNER</div><h1>MA Signals <span style="color:#6ee2b8">20/50</span></h1><div class="muted">Svenska aktier. Tydliga trender. Datadrivna signaler.</div></div>',unsafe_allow_html=True)
with st.expander('⚙️ Justera filter',expanded=False):
    a,b,c=st.columns(3)
    with a:
        min_turn=st.number_input('Min. dagsomsättning (MSEK)',0.,500.,20.,5.)
        min_adx=st.slider('Min. ADX (14)',0,60,25)
        min_rsi,max_rsi=st.slider('RSI (14)',0,100,(50,70))
    with b:
        min_atr,max_atr=st.slider('ATR / kurs (%)',0.,15.,(2.,5.),.5)
        min_rvol=st.number_input('Min. relativ volym',0.,10.,1.2,.1)
        cross_days=st.slider('Ny korsning inom (handelsdagar)',1,20,5)
    with c:
        use_ma200=st.checkbox('Kurs över MA200',True)
        use_ma50=st.checkbox('MA50 över MA200',True)
        require_cross=st.checkbox('Kräv ny MA20/50-korsning',False)
        only_pass=st.checkbox('Visa endast aktier som klarar alla filter',False)
        st.caption('RSI, ADX och ATR använder 14 dagar. Relativ volym jämför dagens volym med föregående 20 dagars medelvärde.')
# Egna aktier lagras i webbadressen så att listan kan bokmärkas och återanvändas.
import re
DEFAULT_CUSTOM = str(st.query_params.get('aktier', ''))
if 'custom_tickers' not in st.session_state:
    st.session_state.custom_tickers = [t for t in DEFAULT_CUSTOM.split(',') if re.fullmatch(r'[A-Z0-9-]{1,24}\.(?:ST|HE|CO|OL)', t)]

st.subheader('Min bevakningslista')
with st.form('add_custom_ticker', clear_on_submit=True):
    c_add, c_btn = st.columns([3, 1], vertical_alignment='bottom')
    with c_add:
        custom_input = st.text_input('Lägg till aktie med börssymbol', placeholder='Exempel: SAND.ST eller VOLV-B.ST')
    with c_btn:
        add_pressed = st.form_submit_button('＋ Lägg till', use_container_width=True)
if add_pressed:
    ticker = custom_input.strip().upper().replace(' ', '')
    if not re.fullmatch(r'[A-Z0-9-]{1,24}\.(?:ST|HE|CO|OL)', ticker):
        st.error('Ange en nordisk Yahoo Finance-symbol, t.ex. SAND.ST, NOKIA.HE, NOVO-B.CO eller EQNR.OL.')
    elif ticker in st.session_state.custom_tickers:
        st.info('Aktien finns redan i din lista.')
    else:
        st.session_state.custom_tickers.append(ticker)
        st.query_params['aktier'] = ','.join(st.session_state.custom_tickers)
        st.success(f'{ticker} tillagd. Aktien kontrolleras när du skannar.')

if st.session_state.custom_tickers:
    st.caption('Egna aktier (ST Sverige · HE Finland · CO Danmark · OL Norge). Spara webbadressen som bokmärke för att behålla listan.')
    for ticker in list(st.session_state.custom_tickers):
        name_col, remove_col = st.columns([4, 1], vertical_alignment='center')
        name_col.write(f'**{ticker}**')
        if remove_col.button('Ta bort', key=f'remove_{ticker}'):
            st.session_state.custom_tickers.remove(ticker)
            if st.session_state.custom_tickers:
                st.query_params['aktier'] = ','.join(st.session_state.custom_tickers)
            elif 'aktier' in st.query_params:
                del st.query_params['aktier']
            st.rerun()

all_tickers = {**TICKERS, **{ticker: ticker for ticker in st.session_state.custom_tickers}}
selected=st.multiselect('Aktier att skanna',list(all_tickers),default=list(all_tickers),help='Välj förvalda och egna nordiska aktier.')
col_run,col_info=st.columns([1,3],vertical_alignment='center')
with col_run: run=st.button('↻  Skanna aktier',type='primary',use_container_width=True)
with col_info:st.caption('Gratis Yahoo Finance-data via yfinance • Kan vara fördröjd eller otillgänglig • Inga påhittade signaler')
if 'scan' not in st.session_state:st.session_state.scan=None
if run:
    if not selected:st.warning('Välj minst en aktie.')
    else:
        results=[];fail=[]
        with st.spinner('Hämtar kurser och beräknar indikatorer…'):
            with ThreadPoolExecutor(max_workers=5) as pool:
                futures={pool.submit(load,all_tickers[name]):name for name in selected}
                for f in as_completed(futures):
                    name=futures[f]
                    try:
                        d=f.result();last=d.iloc[-1];cross=bool(d.CrossUp.tail(cross_days).any());down=bool(d.CrossDown.tail(cross_days).any())
                        checks={'Omsättning':last['Oms MSEK']>min_turn,'ADX':last.ADX>min_adx,'Kurs > MA200':(last.Close>last.MA200) if use_ma200 else True,'MA50 > MA200':(last.MA50>last.MA200) if use_ma50 else True,'ATR %':min_atr<=last['ATR%']<=max_atr,'RSI':min_rsi<=last.RSI<=max_rsi,'RelVol':last.RelVol>min_rvol}
                        passed=sum(bool(x) for x in checks.values());allpass=all(checks.values());signal=('🟢 NY KÖPSIGNAL' if cross and allpass else '🟡 Ny korsning' if cross else '🔴 Nedåtkorsning' if down else '🟢 Upptrend' if last.MA20>last.MA50 else '⚪ Ingen signal')
                        results.append({'Aktie':name,'Ticker':all_tickers[name],'Datum':str(d.index[-1].date()),'Signal':signal,'Filter':f'{passed}/7','Godkänd':allpass,'Ny korsning':cross,'Kurs':round(float(last.Close),2),'MA20':round(float(last.MA20),2),'MA50':round(float(last.MA50),2),'MA200':round(float(last.MA200),2),'ADX':round(float(last.ADX),1),'RSI':round(float(last.RSI),1),'ATR %':round(float(last['ATR%']),2),'ADR %':round(float(last['ADR%']),2),'RelVol':round(float(last.RelVol),2),'Oms MSEK':round(float(last['Oms MSEK']),1),**{f'✓ {k}':bool(v) for k,v in checks.items()}})
                    except Exception as e:fail.append(f'{name}: {e}')
        st.session_state.scan=(results,fail)
if st.session_state.scan is None:
    st.info('Tryck på **Skanna aktier** för att hämta aktuella kurser och visa signalerna. Ingen kursdata har hämtats ännu.')
else:
    results,fail=st.session_state.scan
    if results:
        df=pd.DataFrame(results);df['score']=df['Filter'].str.split('/').str[0].astype(int)
        df=df.sort_values(['Godkänd','Ny korsning','score'],ascending=False)
        shown=df[(df.Godkänd)&((df['Ny korsning']) if require_cross else True)] if only_pass else df[(df['Ny korsning'])] if require_cross else df
        k1,k2,k3,k4=st.columns(4)
        k1.metric('Skannade',len(df));k2.metric('Alla 7 filter',int(df.Godkänd.sum()));k3.metric('Nya korsningar',int(df['Ny korsning'].sum()));k4.metric('Nya köpsignaler',int((df.Godkänd & df['Ny korsning']).sum()))
        st.subheader('Marknadsöversikt')
        st.dataframe(shown.drop(columns=['score','Godkänd','Ny korsning']),hide_index=True,use_container_width=True,height=min(680,100+len(shown)*37),column_config={'Kurs':st.column_config.NumberColumn(format='%.2f kr'),'ATR %':st.column_config.NumberColumn(format='%.2f %%'),'ADR %':st.column_config.NumberColumn(format='%.2f %%')})
        st.download_button('↓ Exportera resultat (CSV)',df.drop(columns=['score']).to_csv(index=False).encode('utf-8-sig'),file_name='nord_ma_signaler.csv',mime='text/csv')
        st.caption('🟢 NY KÖPSIGNAL = ny MA20/50-korsning inom valt fönster OCH alla sju filter uppfyllda. En signal är inte en garanti för avkastning.')
        st.subheader('Aktiedetaljer')
        pick=st.selectbox('Visa graf för',df.Aktie.tolist())
        row=df.loc[df.Aktie==pick].iloc[0]
        d=load(all_tickers[pick]).tail(200)
        fig=go.Figure()
        fig.add_trace(go.Candlestick(x=d.index,open=d.Open,high=d.High,low=d.Low,close=d.Close,name='Kurs',increasing_line_color='#6ee2b8',decreasing_line_color='#ef7185'))
        for n,color in [(20,'#6ee2b8'),(50,'#f2b85d'),(200,'#8e9dbd')]:fig.add_trace(go.Scatter(x=d.index,y=d[f'MA{n}'],mode='lines',name=f'MA{n}',line=dict(color=color,width=2)))
        cross=d[d.CrossUp]
        fig.add_trace(go.Scatter(x=cross.index,y=cross.Close,mode='markers',name='Korsning upp',marker=dict(color='#6ee2b8',symbol='triangle-up',size=12)))
        fig.update_layout(template='plotly_dark',paper_bgcolor='#0a101b',plot_bgcolor='#111b29',font_color='#dbe5f3',height=430,margin=dict(l=5,r=5,t=15,b=5),xaxis_rangeslider_visible=False,legend=dict(orientation='h',y=1.08),xaxis=dict(showgrid=False),yaxis=dict(gridcolor='#263344'))
        st.plotly_chart(fig,use_container_width=True)
        st.caption(f"Senaste datadag: {row['Datum']} · {pick} · {row['Signal']} · {row['Filter']} filter godkända")
    else:st.error('Ingen kursdata kunde hämtas. Försök igen senare.')
    if fail:
        with st.expander(f'⚠️ {len(fail)} aktier kunde inte hämtas'):
            for err in fail:st.write(err)
st.divider()
st.caption('Utbildnings- och analysverktyg, inte investeringsrådgivning. Daglig omsättning approximeras med stängningskurs × volym (20-dagarsmedel). Beräkningarna kan skilja sig från Börsdata/TradingView. Gratis datakällor har ingen garanterad tillgänglighet. Webbappen behöver internet och en Python/Streamlit-server.')
