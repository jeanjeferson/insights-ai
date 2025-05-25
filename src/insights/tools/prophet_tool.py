from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import pandas as pd
from prophet import Prophet
import json
import warnings
from datetime import datetime, timedelta

# Suprimir warnings do Prophet
warnings.filterwarnings('ignore', category=FutureWarning)

class ProphetForecastInput(BaseModel):
    """Schema otimizado para previsões Prophet com validações robustas."""
    
    data_path: str = Field(
        ...,
        description="Caminho para arquivo CSV com dados históricos de vendas. Use 'data/vendas.csv' para dados principais.",
        example="data/vendas.csv"
    )
    
    date_column: str = Field(
        "Data",
        description="Nome da coluna contendo datas. Padrão: 'Data' (formato YYYY-MM-DD).",
        example="Data"
    )
    
    target_column: str = Field(
        "Total_Liquido",
        description="Coluna a ser prevista. Use 'Total_Liquido' para receita, 'Quantidade' para volume de vendas.",
        example="Total_Liquido"
    )
    
    periods: int = Field(
        15,
        description="Dias futuros para previsão (1-90). Use 15 para análise tática, 30-60 para planejamento estratégico.",
        ge=1,
        le=90
    )
    
    aggregation: Optional[str] = Field(
        "daily",
        description="Agregação temporal: 'daily' (diário), 'weekly' (semanal), 'monthly' (mensal).",
        pattern="^(daily|weekly|monthly)$"
    )
    
    seasonality_mode: Optional[str] = Field(
        "multiplicative",
        description="Modo sazonalidade: 'multiplicative' (padrão joalherias), 'additive' (crescimento linear).",
        pattern="^(multiplicative|additive)$"
    )
    
    include_holidays: Optional[bool] = Field(
        True,
        description="Incluir feriados brasileiros (Natal, Dia das Mães, etc.). Recomendado: True para joalherias."
    )
    
    confidence_interval: Optional[float] = Field(
        0.80,
        description="Intervalo de confiança (0.5-0.95). Use 0.80 para análise padrão, 0.95 para cenários conservadores.",
        ge=0.5,
        le=0.95
    )
    
    @validator('target_column')
    def validate_target_column(cls, v):
        allowed_columns = ['Total_Liquido', 'Quantidade', 'Preco_Tabela', 'Custo_Produto']
        if v not in allowed_columns:
            raise ValueError(f"target_column deve ser um de: {allowed_columns}")
        return v

class ProphetForecastTool(BaseTool):
    """
    🔮 FERRAMENTA DE PREVISÃO PROFISSIONAL COM PROPHET
    
    QUANDO USAR:
    - Criar projeções de vendas para planejamento estratégico
    - Prever demanda para gestão de estoque
    - Analisar tendências futuras de receita
    - Modelar impacto de sazonalidade em vendas
    - Validar metas comerciais com base em histórico
    
    CASOS DE USO ESPECÍFICOS:
    - Projeção de vendas para próximos 15-30 dias
    - Previsão de demanda por categoria de produto
    - Análise de impacto de campanhas sazonais
    - Planejamento de estoque baseado em forecast
    - Modelagem de cenários otimista/conservador
    
    RESULTADOS ENTREGUES:
    - Previsões diárias com intervalos de confiança
    - Decomposição de tendência e sazonalidade
    - Métricas de precisão do modelo
    - Insights de negócio e recomendações
    - Análise de riscos e oportunidades
    """
    
    name: str = "Prophet Forecast Tool"
    description: str = (
        "Ferramenta de previsão profissional usando Prophet para análise de séries temporais. "
        "Cria projeções precisas de vendas considerando tendências, sazonalidade e feriados. "
        "Ideal para planejamento estratégico, gestão de estoque e validação de metas comerciais."
    )
    args_schema: Type[BaseModel] = ProphetForecastInput
    
    def _run(
        self,
        data_path: str,
        date_column: str = "Data",
        target_column: str = "Total_Liquido",
        periods: int = 15,
        aggregation: str = "daily",
        seasonality_mode: str = "multiplicative",
        include_holidays: bool = True,
        confidence_interval: float = 0.80
    ) -> str:
        """
        Executa previsão Prophet com configurações otimizadas para joalherias.
        
        Returns:
            JSON estruturado com previsões, métricas e insights de negócio
        """
        try:
            print(f"🔮 Iniciando previsão Prophet para {target_column}")
            print(f"📅 Períodos: {periods} dias | Agregação: {aggregation}")
            
            # Carregar e preparar dados
            df = pd.read_csv(data_path, sep=';', encoding='utf-8')
            
            if date_column not in df.columns:
                raise ValueError(f"Coluna '{date_column}' não encontrada. Colunas disponíveis: {list(df.columns)}")
            
            if target_column not in df.columns:
                raise ValueError(f"Coluna '{target_column}' não encontrada. Colunas disponíveis: {list(df.columns)}")
            
            # Preparar dados para Prophet
            prophet_df = self._prepare_data_for_prophet(df, date_column, target_column, aggregation)
            
            # Configurar e treinar modelo
            model = self._configure_prophet_model(
                seasonality_mode=seasonality_mode,
                include_holidays=include_holidays,
                confidence_interval=confidence_interval
            )
            
            print("🤖 Treinando modelo Prophet...")
            model.fit(prophet_df)
            
            # Criar previsões
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
            
            # Calcular métricas de precisão
            metrics = self._calculate_model_metrics(prophet_df, forecast)
            
            # Extrair insights de negócio
            business_insights = self._extract_business_insights(forecast, target_column, periods)
            
            # Estruturar resultados
            results = {
                "forecast_summary": {
                    "target_metric": target_column,
                    "forecast_periods": periods,
                    "model_accuracy": metrics,
                    "last_actual_value": float(prophet_df['y'].iloc[-1]),
                    "forecast_trend": business_insights["trend_direction"]
                },
                "predictions": self._format_predictions(forecast, periods, date_column, target_column),
                "business_insights": business_insights,
                "model_components": self._extract_model_components(forecast),
                "recommendations": self._generate_business_recommendations(business_insights, target_column),
                "metadata": {
                    "model_type": "prophet",
                    "seasonality_mode": seasonality_mode,
                    "confidence_interval": confidence_interval,
                    "include_holidays": include_holidays,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            print("✅ Previsão Prophet concluída com sucesso")
            return json.dumps(results, ensure_ascii=False, indent=2, default=str)
            
        except Exception as e:
            error_response = {
                "error": f"Erro na previsão Prophet: {str(e)}",
                "target_column": target_column,
                "data_path": data_path,
                "troubleshooting": {
                    "check_file_exists": f"Verifique se {data_path} existe",
                    "check_columns": f"Confirme se colunas '{date_column}' e '{target_column}' existem",
                    "check_data_format": "Dados devem ter pelo menos 30 registros históricos"
                },
                "metadata": {
                    "status": "error",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)
    
    def _prepare_data_for_prophet(self, df: pd.DataFrame, date_col: str, target_col: str, aggregation: str) -> pd.DataFrame:
        """Preparar dados no formato Prophet (ds, y)"""
        # Converter data
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Agregar dados conforme especificado
        if aggregation == "daily":
            grouped = df.groupby(date_col)[target_col].sum().reset_index()
        elif aggregation == "weekly":
            df['week'] = df[date_col].dt.to_period('W').dt.start_time
            grouped = df.groupby('week')[target_col].sum().reset_index()
            grouped = grouped.rename(columns={'week': date_col})
        elif aggregation == "monthly":
            df['month'] = df[date_col].dt.to_period('M').dt.start_time
            grouped = df.groupby('month')[target_col].sum().reset_index()
            grouped = grouped.rename(columns={'month': date_col})
        
        # Formato Prophet
        prophet_df = grouped.rename(columns={date_col: 'ds', target_col: 'y'})
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Remover valores nulos
        prophet_df = prophet_df.dropna()
        
        return prophet_df
    
    def _configure_prophet_model(self, seasonality_mode: str, include_holidays: bool, confidence_interval: float) -> Prophet:
        """Configurar modelo Prophet otimizado para joalherias"""
        model = Prophet(
            seasonality_mode=seasonality_mode,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=confidence_interval,
            changepoint_prior_scale=0.05,  # Sensibilidade a mudanças de tendência
            seasonality_prior_scale=10.0,  # Força da sazonalidade
        )
        
        # Adicionar feriados brasileiros se solicitado
        if include_holidays:
            holidays = self._create_brazilian_holidays()
            model.add_country_holidays(country_name='BR')
            
            # Feriados específicos para joalherias
            jewelry_holidays = pd.DataFrame({
                'holiday': ['Dia das Mães', 'Dia dos Namorados', 'Black Friday', 'Natal'],
                'ds': pd.to_datetime(['2024-05-12', '2024-06-12', '2024-11-29', '2024-12-25']),
                'lower_window': [-7, -7, -3, -15],
                'upper_window': [1, 1, 1, 1],
            })
            
            for _, row in jewelry_holidays.iterrows():
                model.add_holiday(
                    name=row['holiday'],
                    dates=pd.to_datetime([row['ds']]),
                    lower_window=row['lower_window'],
                    upper_window=row['upper_window']
                )
        
        return model
    
    def _create_brazilian_holidays(self) -> pd.DataFrame:
        """Criar DataFrame com feriados brasileiros relevantes"""
        holidays = pd.DataFrame({
            'holiday': ['Ano Novo', 'Carnaval', 'Páscoa', 'Dia do Trabalho', 'Independência', 'Nossa Senhora', 'Finados', 'Proclamação'],
            'ds': pd.to_datetime([
                '2024-01-01', '2024-02-13', '2024-03-31', '2024-05-01', 
                '2024-09-07', '2024-10-12', '2024-11-02', '2024-11-15'
            ])
        })
        return holidays
    
    def _calculate_model_metrics(self, historical_data: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, float]:
        """Calcular métricas de precisão do modelo"""
        # Pegar apenas dados históricos para validação
        historical_forecast = forecast.iloc[:len(historical_data)]
        
        actual = historical_data['y'].values
        predicted = historical_forecast['yhat'].values
        
        # Calcular métricas
        mae = abs(actual - predicted).mean()
        mape = (abs((actual - predicted) / actual) * 100).mean()
        rmse = ((actual - predicted) ** 2).mean() ** 0.5
        
        return {
            "mae": round(float(mae), 2),
            "mape": round(float(mape), 2),
            "rmse": round(float(rmse), 2),
            "accuracy_score": round(100 - float(mape), 2)
        }
    
    def _extract_business_insights(self, forecast: pd.DataFrame, target_column: str, periods: int) -> Dict[str, Any]:
        """Extrair insights de negócio das previsões"""
        future_forecast = forecast.tail(periods)
        historical_avg = forecast.head(-periods)['yhat'].mean()
        future_avg = future_forecast['yhat'].mean()
        
        # Calcular tendência
        trend_change = ((future_avg - historical_avg) / historical_avg) * 100
        
        # Identificar picos e vales
        max_day = future_forecast.loc[future_forecast['yhat'].idxmax()]
        min_day = future_forecast.loc[future_forecast['yhat'].idxmin()]
        
        return {
            "trend_direction": "crescimento" if trend_change > 5 else "declínio" if trend_change < -5 else "estável",
            "trend_percentage": round(trend_change, 2),
            "future_average": round(future_avg, 2),
            "historical_average": round(historical_avg, 2),
            "peak_day": {
                "date": max_day['ds'].strftime("%Y-%m-%d"),
                "value": round(max_day['yhat'], 2),
                "confidence_upper": round(max_day['yhat_upper'], 2)
            },
            "lowest_day": {
                "date": min_day['ds'].strftime("%Y-%m-%d"),
                "value": round(min_day['yhat'], 2),
                "confidence_lower": round(min_day['yhat_lower'], 2)
            },
            "total_forecast": round(future_forecast['yhat'].sum(), 2),
            "volatility": round(future_forecast['yhat'].std(), 2)
        }
    
    def _format_predictions(self, forecast: pd.DataFrame, periods: int, date_col: str, target_col: str) -> list:
        """Formatar previsões para consumo fácil"""
        future_forecast = forecast.tail(periods)
        
        predictions = []
        for _, row in future_forecast.iterrows():
            predictions.append({
                "date": row['ds'].strftime("%Y-%m-%d"),
                "predicted_value": round(row['yhat'], 2),
                "lower_bound": round(row['yhat_lower'], 2),
                "upper_bound": round(row['yhat_upper'], 2),
                "confidence": "high" if abs(row['yhat_upper'] - row['yhat_lower']) < row['yhat'] * 0.3 else "medium"
            })
        
        return predictions
    
    def _extract_model_components(self, forecast: pd.DataFrame) -> Dict[str, Any]:
        """Extrair componentes do modelo (tendência, sazonalidade)"""
        return {
            "trend_strength": "strong" if forecast['trend'].std() > forecast['trend'].mean() * 0.1 else "moderate",
            "seasonal_pattern": "high" if 'yearly' in forecast.columns and forecast['yearly'].std() > 0 else "low",
            "weekly_pattern": "significant" if 'weekly' in forecast.columns and forecast['weekly'].std() > 0 else "minimal"
        }
    
    def _generate_business_recommendations(self, insights: Dict[str, Any], target_column: str) -> list:
        """Gerar recomendações de negócio baseadas nas previsões"""
        recommendations = []
        
        if insights["trend_direction"] == "crescimento":
            recommendations.append(f"📈 Tendência de crescimento de {insights['trend_percentage']:.1f}% - considere aumentar estoque")
            recommendations.append("🎯 Oportunidade para campanhas promocionais agressivas")
        elif insights["trend_direction"] == "declínio":
            recommendations.append(f"📉 Tendência de declínio de {insights['trend_percentage']:.1f}% - revisar estratégia")
            recommendations.append("⚠️ Considere promoções para estimular demanda")
        
        # Recomendações específicas por métrica
        if target_column == "Total_Liquido":
            recommendations.append(f"💰 Receita prevista: R$ {insights['total_forecast']:,.2f}")
            recommendations.append(f"📊 Pico esperado em {insights['peak_day']['date']}")
        elif target_column == "Quantidade":
            recommendations.append(f"📦 Volume previsto: {insights['total_forecast']:,.0f} unidades")
            recommendations.append("🏪 Ajustar níveis de estoque conforme previsão")
        
        # Recomendação sobre volatilidade
        if insights["volatility"] > insights["future_average"] * 0.2:
            recommendations.append("⚡ Alta volatilidade prevista - monitorar de perto")
        
        return recommendations