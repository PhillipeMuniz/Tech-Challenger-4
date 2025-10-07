import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import locale
import streamlit_shadcn_ui as ui
from datetime import datetime, timedelta

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
st.set_page_config(layout='wide')

symbols = pd.read_excel("symbols.xlsx")

empresa_para_codigo = dict(zip(symbols['Company'], symbols['Symbol']))
opcoes_empresas = symbols['Company'].tolist()
empresas_selecionadas = st.sidebar.multiselect(
    "Selecione uma ou mais Empresas (máximo 2)", opcoes_empresas)

if len(empresas_selecionadas) > 2:
    st.sidebar.error("Por favor, selecione no máximo 2 empresas.")
    empresas_selecionadas = empresas_selecionadas[:2]


periodo_map = {
    '1d': '1 dia',
    '5d': '5 dias',
    '1mo': '1 mês',
    '3mo': '3 meses',
    '6mo': '6 meses',
    '1y': '1 ano',
    '2y': '2 anos',
    '5y': '5 anos',
    '10y': '10 anos',
    'ytd': 'Ano até a data',
    'max': 'Máximo'
}

descricao_para_periodo = {v: k for k, v in periodo_map.items()}
opcoes_periodo_legivel = list(periodo_map.values())
periodo_selecionado_legivel = st.sidebar.selectbox("Selecione o Período", opcoes_periodo_legivel)
periodo_selecionado = descricao_para_periodo[periodo_selecionado_legivel]


botao_pesquisar = st.sidebar.button("Pesquisar")


if botao_pesquisar:
    st.title("Comparação de Empresas - NYSE")

    codigos_selecionados = [empresa_para_codigo[empresa] for empresa in empresas_selecionadas]

    if codigos_selecionados:
        tickers = yf.Tickers(' '.join(codigos_selecionados))

        if len(codigos_selecionados) > 0:
            cols = st.columns(len(codigos_selecionados))
            for i, ticker in enumerate(codigos_selecionados):
                dados = tickers.tickers[ticker]
                info = dados.info

                hist = dados.history(period=periodo_selecionado)

                if not hist.empty:
                    preco_inicio = hist['Close'].iloc[0]
                    preco_fim = hist['Close'].iloc[-1]
                    variacao_percentual = ((preco_fim - preco_inicio) / preco_inicio) * 100
                else:
                    variacao_percentual = 0.0

                with cols[i]:
                    ui.metric_card(
                        title="Preço Atual",
                        content=f"R$ {info.get('currentPrice', 'Não disponível')}",
                        description="",
                        key=f"card1_{i}"
                    )
                    
                    ui.metric_card(
                        title="Variação Percentual (Último Período)",
                        content=f"{variacao_percentual:.2f}%",
                        description="Variação percentual do preço de fechamento ao longo do período selecionado.",
                        key=f"card2_{i}"
                    )

        cols_graficos = st.columns(len(codigos_selecionados))
        cols_info = st.columns(len(codigos_selecionados))

        for i, ticker in enumerate(codigos_selecionados):
            dados = tickers.tickers[ticker]
            info = dados.info
            
            hist = dados.history(period=periodo_selecionado)
            
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()

            fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                                 open=hist['Open'],
                                                 high=hist['High'],
                                                 low=hist['Low'],
                                                 close=hist['Close'])])
            
            fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], mode='lines', name='SMA 20'))
            
            fig.update_layout(title=f'{info.get("shortName", "Nome não disponível")} - {periodo_selecionado_legivel}',
                              xaxis_title='Data',
                              yaxis_title='Preço',
                              xaxis_rangeslider_visible=False)
            
            with cols_graficos[i]:
                st.plotly_chart(fig, use_container_width=True)

            indicadores = {
                "Indicadores": {
                    "Dividend Yield": f"{info.get('dividendRate', 0) / info.get('currentPrice', 1):.2%}",
                    "Preço/Lucro": f"{info.get('currentPrice', 1) / (info.get('netIncomeToCommon', 1) / info.get('sharesOutstanding', 1)):.2f}",
                    "Preço/Valor Patrimonial": f"{info.get('currentPrice', 1) / info.get('bookValue', 1):.2f}",
                    "Preço/EBITDA": f"{info.get('currentPrice', 1) / (info.get('ebitda', 1) / info.get('sharesOutstanding', 1)):.2f}",
                    "ROE": f"{(info.get('netIncomeToCommon', 0) / info.get('marketCap', 1)) * 100:.2f}%",
                    "ROA": f"{(info.get('netIncomeToCommon', 0) / info.get('marketCap', 1)) * 100:.2f}%",
                    "Margem Bruta": f"{info.get('grossMargins', 0) * 100:.2f}%",
                    "Margem EBITDA": f"{(info.get('ebitda', 1) / info.get('totalRevenue', 1)) * 100:.2f}%",
                    "Último Dividendo": f"{info.get('lastDividendValue', 0):.2f}",
                    "Dívida Líquida/EBITDA": f"{(info.get('totalDebt', 0) - info.get('totalCash', 0)) / info.get('ebitda', 1):.2f}"    
                },
                "Informações da Empresa": {
                    "Site": info.get('website', 'Não disponível'),
                    "Setor": info.get('sector', 'Não disponível'),
                    "Indústria": info.get('industry', 'Não disponível'),
                }
            }

            with cols_info[i]:
                for secao, valores in indicadores.items():
                    st.write(f"### {secao}")
                    for indicador, valor in valores.items():
                        st.write(f"- **{indicador}:** {valor}")

                st.write("### Descrição dos Dados Históricos")
                st.write(hist.describe())

    else:
        st.sidebar.error("Nenhuma empresa selecionada.")


