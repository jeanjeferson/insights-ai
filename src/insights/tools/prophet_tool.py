from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64

class ProphetForecastInput(BaseModel):
    """Schema de entrada para a ferramenta de previsão com Prophet."""
    data: str = Field(..., description="JSON string contendo o DataFrame de vendas")
    data_column: str = Field(..., description="Nome da coluna contendo as datas")
    target_column: str = Field(..., description="Nome da coluna contendo os valores a serem previstos")
    periods: int = Field(15, description="Número de períodos futuros para previsão")
    include_history: bool = Field(True, description="Incluir dados históricos na previsão")
    seasonality_mode: str = Field("multiplicative", description="Modo de sazonalidade: 'multiplicative' ou 'additive'")
    daily_seasonality: bool = Field(True, description="Incluir sazonalidade diária")
    weekly_seasonality: bool = Field(True, description="Incluir sazonalidade semanal")
    yearly_seasonality: bool = Field(True, description="Incluir sazonalidade anual")
    
class ProphetForecastTool(BaseTool):
    name: str = "Prophet Forecast Tool"
    description: str = (
        "Realiza previsões de séries temporais utilizando o modelo Prophet do Facebook. "
        "Fornece projeções futuras baseadas em dados históricos, considerando sazonalidade e tendências."
    )
    args_schema: Type[BaseModel] = ProphetForecastInput
    
    def _run(
        self, 
        data: str,
        data_column: str,
        target_column: str,
        periods: int = 15,
        include_history: bool = True,
        seasonality_mode: str = "multiplicative",
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True
    ) -> str:
        try:
            # Converter string JSON para DataFrame
            df = pd.read_json(data)
            
            # Verificar se as colunas necessárias existem
            if data_column not in df.columns:
                return f"Erro: Coluna de data '{data_column}' não encontrada no DataFrame."
            if target_column not in df.columns:
                return f"Erro: Coluna alvo '{target_column}' não encontrada no DataFrame."
            
            # Preparar dados para o Prophet (requer colunas 'ds' e 'y')
            prophet_df = df[[data_column, target_column]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Converter coluna de data para datetime se ainda não for
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Criar e treinar modelo Prophet
            model = Prophet(
                seasonality_mode=seasonality_mode,
                daily_seasonality=daily_seasonality,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality
            )
            
            # Adicionar feriados brasileiros se necessário
            # model.add_country_holidays(country_name='BR')
            
            # Treinar o modelo
            model.fit(prophet_df)
            
            # Criar DataFrame para previsão futura
            future = model.make_future_dataframe(periods=periods, freq='D')
            
            # Realizar previsão
            forecast = model.predict(future)
            
            # Criar gráficos
            fig1 = model.plot(forecast)
            fig2 = model.plot_components(forecast)
            
            # Converter gráficos para base64 para retornar
            buf = io.BytesIO()
            fig1.savefig(buf, format='png')
            buf.seek(0)
            img1 = base64.b64encode(buf.read()).decode('utf-8')
            
            buf = io.BytesIO()
            fig2.savefig(buf, format='png')
            buf.seek(0)
            img2 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Preparar resultados
            if include_history:
                result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            else:
                result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[-periods:]
            
            # Converter de volta para o formato original de data
            result_df = result_df.rename(columns={'ds': data_column, 'yhat': f'previsao_{target_column}', 
                                                 'yhat_lower': f'limite_inferior_{target_column}', 
                                                 'yhat_upper': f'limite_superior_{target_column}'})
            
            # Retornar resultados em formato JSON
            results = {
                'forecast_data': result_df.to_json(orient='records', date_format='iso'),
                'plot': img1,
                'components_plot': img2,
                'model_params': {
                    'seasonality_mode': seasonality_mode,
                    'daily_seasonality': daily_seasonality,
                    'weekly_seasonality': weekly_seasonality,
                    'yearly_seasonality': yearly_seasonality
                }
            }
            
            return str(results)
            
        except Exception as e:
            return f"Erro ao executar previsão com Prophet: {str(e)}"