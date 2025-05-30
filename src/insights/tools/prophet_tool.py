from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
import pandas as pd
from prophet import Prophet
import json
import warnings
from datetime import datetime, timedelta
import os
import time
import traceback
import sys

# Suprimir warnings do Prophet
warnings.filterwarnings('ignore', category=FutureWarning)

class ProphetForecastInput(BaseModel):
    """Schema otimizado para previsões Prophet com validações robustas."""
    
    data_path: str = Field(
        ...,
        description="Caminho para arquivo CSV com dados históricos de vendas. Use 'data/vendas.csv' para dados principais.",
        json_schema_extra={"example": "data/vendas.csv"}
    )
    
    date_column: str = Field(
        "Data",
        description="Nome da coluna contendo datas. Padrão: 'Data' (formato YYYY-MM-DD).",
        json_schema_extra={"example": "Data"}
    )
    
    target_column: str = Field(
        "Total_Liquido",
        description="Coluna a ser prevista. Use 'Total_Liquido' para receita, 'Quantidade' para volume de vendas.",
        json_schema_extra={"example": "Total_Liquido"}
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
        json_schema_extra={
            "pattern": "^(daily|weekly|monthly)$"
        }
    )
    
    seasonality_mode: Optional[str] = Field(
        "multiplicative",
        description="Modo sazonalidade: 'multiplicative' (padrão joalherias), 'additive' (crescimento linear).",
        json_schema_extra={
            "pattern": "^(multiplicative|additive)$"
        }
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
    
    @field_validator('target_column')
    @classmethod
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
            
            # Carregar e preparar dados com tratamento de tipos
            df = pd.read_csv(
                data_path, 
                sep=';', 
                encoding='utf-8',
                low_memory=False,
                dtype={
                    'Codigo_Produto': 'str',
                    'Codigo_Cliente': 'str',
                    'Grupo_Produto': 'str',
                    'Descricao': 'str',
                    'Quantidade': 'float64',
                    'Total_Liquido': 'float64'
                }
            )
            
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
        try:
            # Configuração básica do Prophet com tratamento de compatibilidade
            prophet_config = {
                'seasonality_mode': seasonality_mode,
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'interval_width': confidence_interval,
                'changepoint_prior_scale': 0.05,  # Sensibilidade a mudanças de tendência
                'seasonality_prior_scale': 10.0,  # Força da sazonalidade
            }
            
            # Tentar diferentes configurações dependendo da versão do Prophet
            try:
                # Versão mais recente com stan_backend
                model = Prophet(stan_backend=None, **prophet_config)
            except TypeError:
                try:
                    # Versão intermediária
                    model = Prophet(**prophet_config)
                except Exception as e:
                    # Fallback para configuração mínima
                    print(f"⚠️ Usando configuração mínima do Prophet: {str(e)}")
                    model = Prophet(
                        seasonality_mode=seasonality_mode,
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True,
                        interval_width=confidence_interval
                    )
            
            # Adicionar feriados brasileiros se solicitado
            if include_holidays:
                try:
                    # Usar feriados brasileiros diretamente
                    model.add_country_holidays(country_name='BR')
                    
                    # Feriados específicos para joalherias como sazonalidades customizadas
                    jewelry_holidays = pd.DataFrame({
                        'holiday': ['Dia das Mães', 'Dia dos Namorados', 'Black Friday', 'Natal'],
                        'ds': pd.to_datetime(['2024-05-12', '2024-06-12', '2024-11-29', '2024-12-25']),
                        'lower_window': [-7, -7, -3, -15],
                        'upper_window': [1, 1, 1, 1],
                    })
                    
                    # Adicionar como eventos/feriados usando o método correto
                    for _, row in jewelry_holidays.iterrows():
                        try:
                            # Usar a API correta do Prophet para feriados
                            if hasattr(model, 'holidays'):
                                if model.holidays is None:
                                    model.holidays = pd.DataFrame(columns=['ds', 'holiday'])
                                
                                new_holiday = pd.DataFrame({
                                    'ds': [row['ds']],
                                    'holiday': [row['holiday']]
                                })
                                model.holidays = pd.concat([model.holidays, new_holiday], ignore_index=True)
                            else:
                                # Método alternativo para versões mais antigas
                                pass
                        except Exception as e:
                            print(f"⚠️ Aviso: Não foi possível adicionar feriado {row['holiday']}: {str(e)}")
                            
                except Exception as e:
                    print(f"⚠️ Aviso: Não foi possível configurar feriados brasileiros: {str(e)}")
            
            return model
            
        except Exception as e:
            print(f"❌ Erro na configuração do Prophet: {str(e)}")
            # Fallback para configuração mais simples
            return Prophet(
                seasonality_mode='additive',
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.8
            )
    
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

    def generate_prophet_test_report(self, test_data: dict) -> str:
        """Gera relatório visual completo dos testes Prophet em formato markdown."""
        
        # Coletar dados com fallbacks
        metadata = test_data.get('metadata', {})
        data_metrics = test_data.get('data_metrics', {})
        results = test_data.get('results', {})
        component_tests = test_data.get('component_tests', {})
        
        report = [
            "# 🔮 Teste Completo de Previsões Prophet - Relatório Executivo",
            f"**Data do Teste:** {metadata.get('test_timestamp', 'N/A')}",
            f"**Fonte de Dados:** `{metadata.get('data_source', 'desconhecida')}`",
            f"**Registros Analisados:** {data_metrics.get('total_records', 0):,}",
            f"**Modelos Testados:** {data_metrics.get('models_tested', 0):,}",
            f"**Intervalo de Dados:** {data_metrics.get('date_range', {}).get('start', 'N/A')} até {data_metrics.get('date_range', {}).get('end', 'N/A')}",
            "\n## 📈 Performance de Execução",
            f"```\n{json.dumps(test_data.get('performance_metrics', {}), indent=2)}\n```",
            "\n## 🎯 Resumo dos Testes Executados"
        ]
        
        # Contabilizar sucessos e falhas
        successful_tests = len([r for r in results.values() if 'success' in r and r['success']])
        failed_tests = len([r for r in results.values() if 'success' in r and not r['success']])
        total_tests = len(results)
        
        report.extend([
            f"- **Total de Componentes:** {total_tests}",
            f"- **Sucessos:** {successful_tests} ✅",
            f"- **Falhas:** {failed_tests} ❌",
            f"- **Taxa de Sucesso:** {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "- **Taxa de Sucesso:** N/A"
        ])
        
        # Principais Descobertas das Previsões
        report.append("\n## 🔮 Principais Descobertas das Previsões")
        
        # Previsões de Receita
        if 'revenue_forecast' in results and results['revenue_forecast'].get('success'):
            revenue_data = results['revenue_forecast']
            accuracy = revenue_data.get('accuracy_score', 0)
            trend = revenue_data.get('trend_direction', 'N/A')
            total_forecast = revenue_data.get('total_forecast', 0)
            report.append(f"- **Previsão de Receita (15 dias):** R$ {total_forecast:,.0f}")
            report.append(f"- **Precisão do Modelo:** {accuracy:.1f}%")
            report.append(f"- **Tendência Identificada:** {trend}")
        
        # Previsões de Volume
        if 'volume_forecast' in results and results['volume_forecast'].get('success'):
            volume_data = results['volume_forecast']
            volume_forecast = volume_data.get('total_forecast', 0)
            volume_accuracy = volume_data.get('accuracy_score', 0)
            report.append(f"- **Previsão de Volume (15 dias):** {volume_forecast:,.0f} unidades")
            report.append(f"- **Precisão do Volume:** {volume_accuracy:.1f}%")
        
        # Análise de Agregações
        if 'aggregation_analysis' in results:
            agg_data = results['aggregation_analysis']
            best_aggregation = agg_data.get('best_performing', 'N/A')
            report.append(f"- **Melhor Agregação Temporal:** {best_aggregation}")
        
        # Análise de Sazonalidade
        if 'seasonality_analysis' in results:
            season_data = results['seasonality_analysis']
            seasonal_strength = season_data.get('seasonal_strength', 'N/A')
            report.append(f"- **Força da Sazonalidade:** {seasonal_strength}")
        
        # Detalhamento por Componente
        report.append("\n## 🔧 Detalhamento dos Componentes Testados")
        
        component_categories = {
            'Preparação de Dados': ['data_loading', 'data_preparation'],
            'Modelos de Receita': ['revenue_forecast', 'revenue_15d', 'revenue_30d'],
            'Modelos de Volume': ['volume_forecast', 'volume_15d', 'volume_30d'],
            'Agregações Temporais': ['daily_aggregation', 'weekly_aggregation', 'monthly_aggregation'],
            'Configurações de Sazonalidade': ['multiplicative_seasonality', 'additive_seasonality'],
            'Análise de Feriados': ['with_holidays', 'without_holidays'],
            'Validação de Modelos': ['model_validation', 'accuracy_assessment']
        }
        
        for category, components in component_categories.items():
            report.append(f"\n### {category}")
            for component in components:
                if component in results:
                    if results[component].get('success'):
                        metrics = results[component].get('metrics', {})
                        report.append(f"- ✅ **{component}**: Concluído")
                        if 'processing_time' in metrics:
                            report.append(f"  - Tempo: {metrics['processing_time']:.3f}s")
                        if 'accuracy_score' in results[component]:
                            report.append(f"  - Precisão: {results[component]['accuracy_score']:.1f}%")
                    else:
                        error_msg = results[component].get('error', 'Erro desconhecido')
                        report.append(f"- ❌ **{component}**: {error_msg}")
                else:
                    report.append(f"- ⏭️ **{component}**: Não testado")
        
        # Análise de Configurações
        report.append("\n## ⚙️ Teste de Configurações")
        
        if 'configuration_tests' in component_tests:
            config_tests = component_tests['configuration_tests']
            for config_name, config_result in config_tests.items():
                status = "✅" if config_result.get('success') else "❌"
                report.append(f"- {status} **{config_name}**: {config_result.get('description', 'N/A')}")
                if 'accuracy' in config_result:
                    report.append(f"  - Precisão: {config_result['accuracy']:.1f}%")
        
        # Qualidade dos Dados e Limitações
        report.append("\n## ⚠️ Qualidade dos Dados e Limitações")
        
        data_quality = data_metrics.get('data_quality_check', {})
        if data_quality:
            report.append("### Qualidade dos Dados:")
            for check, value in data_quality.items():
                if value > 0:
                    report.append(f"- **{check}**: {value} ocorrências")
        
        # Insights de Previsão
        if 'forecast_insights' in component_tests:
            insights = component_tests['forecast_insights']
            report.append(f"\n### Insights de Previsão:")
            for insight_key, insight_value in insights.items():
                if isinstance(insight_value, (int, float)):
                    report.append(f"- **{insight_key}**: {insight_value:,.2f}")
                else:
                    report.append(f"- **{insight_key}**: {insight_value}")
        
        # Recomendações Finais
        report.append("\n## 💡 Recomendações do Sistema de Previsões")
        
        recommendations = [
            "🔮 Utilizar modelos com precisão > 80% para decisões estratégicas",
            "📊 Monitorar tendências identificadas para ajuste de estoque",
            "📅 Considerar sazonalidade em planejamentos de curto prazo",
            "⚡ Recalibrar modelos semanalmente com novos dados",
            "🎯 Focar em previsões de 15-30 dias para maior confiabilidade"
        ]
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        # Erros encontrados
        errors = test_data.get('errors', [])
        if errors:
            report.append(f"\n### Erros Detectados ({len(errors)}):")
            for error in errors[-3:]:  # Últimos 3 erros
                report.append(f"- **{error['context']}**: {error['error_message']}")
        
        return "\n".join(report)

    def run_full_prophet_test(self) -> str:
        """Executa teste completo e retorna relatório formatado"""
        test_result = self.test_all_prophet_components()
        parsed = json.loads(test_result)
        return self.generate_prophet_test_report(parsed)

    def test_all_prophet_components(self, sample_data: str = "data/vendas.csv") -> str:
        """
        Executa teste completo de todos os componentes da classe ProphetForecastTool
        usando especificamente o arquivo data/vendas.csv
        """
        
        # Corrigir caminho do arquivo para usar data/vendas.csv especificamente
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        
        # Usar especificamente data/vendas.csv
        data_file_path = os.path.join(project_root, "data", "vendas.csv")
        
        print(f"🔍 DEBUG: Caminho calculado: {data_file_path}")
        print(f"🔍 DEBUG: Arquivo existe? {os.path.exists(data_file_path)}")
        
        # Verificar se arquivo existe
        if not os.path.exists(data_file_path):
            # Tentar caminhos alternativos
            alternative_paths = [
                os.path.join(project_root, "data", "vendas.csv"),
                os.path.join(os.getcwd(), "data", "vendas.csv"),
                "data/vendas.csv",
                "data\\vendas.csv"
            ]
            
            for alt_path in alternative_paths:
                print(f"🔍 Tentando: {alt_path}")
                if os.path.exists(alt_path):
                    data_file_path = alt_path
                    print(f"✅ Arquivo encontrado em: {data_file_path}")
                    break
            else:
                return json.dumps({
                    "error": f"Arquivo data/vendas.csv não encontrado em nenhum dos caminhos testados",
                    "tested_paths": alternative_paths,
                    "current_dir": current_dir,
                    "project_root": project_root,
                    "working_directory": os.getcwd()
                }, indent=2)

        test_report = {
            "metadata": {
                "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_version": "Prophet Test Suite v1.0",
                "data_source": data_file_path,
                "data_file_specified": "data/vendas.csv",
                "tool_version": "Prophet Forecast Tool v1.0",
                "status": "in_progress"
            },
            "data_metrics": {
                "total_records": 0,
                "models_tested": 0,
                "date_range": {},
                "data_quality_check": {}
            },
            "results": {},
            "component_tests": {},
            "performance_metrics": {},
            "errors": []
        }

        try:
            # 1. Fase de Carregamento de Dados
            test_report["metadata"]["current_stage"] = "data_loading"
            print("\n=== ETAPA 1: CARREGAMENTO DE DADOS PARA PROPHET ===")
            print(f"📁 Carregando especificamente: data/vendas.csv")
            print(f"📁 Caminho completo: {data_file_path}")
            
            start_time = time.time()
            df = pd.read_csv(
                data_file_path, 
                sep=';', 
                encoding='utf-8',
                low_memory=False,
                dtype={
                    'Codigo_Produto': 'str',
                    'Codigo_Cliente': 'str',
                    'Grupo_Produto': 'str',
                    'Descricao': 'str',
                    'Quantidade': 'float64',
                    'Total_Liquido': 'float64'
                }
            )
            loading_time = time.time() - start_time
            
            if df.empty:
                raise Exception("Falha no carregamento do arquivo data/vendas.csv")
            
            print(f"✅ data/vendas.csv carregado: {len(df)} registros em {loading_time:.3f}s")
            
            # Coletar métricas básicas dos dados
            test_report["data_metrics"] = {
                "total_records": int(len(df)),
                "date_range": {
                    "start": str(df['Data'].min()) if 'Data' in df.columns else "N/A",
                    "end": str(df['Data'].max()) if 'Data' in df.columns else "N/A"
                },
                "data_quality_check": self._perform_prophet_data_quality_check(df)
            }
            
            test_report["results"]["data_loading"] = {
                "success": True,
                "metrics": {
                    "processing_time": loading_time,
                    "records_processed": len(df)
                }
            }

            # 2. Teste de Preparação de Dados
            test_report["metadata"]["current_stage"] = "data_preparation"
            print("\n=== ETAPA 2: TESTE DE PREPARAÇÃO DE DADOS ===")
            
            try:
                start_time = time.time()
                print("📊 Testando preparação de dados para Prophet...")
                
                # Testar diferentes agregações
                daily_data = self._prepare_data_for_prophet(df, 'Data', 'Total_Liquido', 'daily')
                weekly_data = self._prepare_data_for_prophet(df, 'Data', 'Total_Liquido', 'weekly')
                monthly_data = self._prepare_data_for_prophet(df, 'Data', 'Total_Liquido', 'monthly')
                
                prep_time = time.time() - start_time
                
                test_report["results"]["data_preparation"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": prep_time,
                        "daily_records": len(daily_data),
                        "weekly_records": len(weekly_data),
                        "monthly_records": len(monthly_data)
                    }
                }
                print(f"✅ Dados preparados: {len(daily_data)} diários, {len(weekly_data)} semanais, {len(monthly_data)} mensais em {prep_time:.3f}s")
                
            except Exception as e:
                self._log_prophet_test_error(test_report, e, "data_preparation")
                print(f"❌ Erro na preparação: {str(e)}")

            # 3. Teste de Previsão de Receita (15 dias)
            test_report["metadata"]["current_stage"] = "revenue_forecast_15d"
            print("\n=== ETAPA 3: TESTE DE PREVISÃO DE RECEITA (15 DIAS) ===")
            
            try:
                start_time = time.time()
                print("💰 Testando previsão de receita para 15 dias...")
                
                revenue_result = self._run(
                    data_path=data_file_path,
                    target_column="Total_Liquido",
                    periods=15,
                    aggregation="daily",
                    seasonality_mode="multiplicative",
                    include_holidays=True
                )
                revenue_time = time.time() - start_time
                
                revenue_data = json.loads(revenue_result)
                if "error" not in revenue_data:
                    accuracy = revenue_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                    trend = revenue_data["business_insights"]["trend_direction"]
                    total_forecast = revenue_data["business_insights"]["total_forecast"]
                    
                    test_report["results"]["revenue_forecast"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": revenue_time,
                            "periods_forecasted": 15
                        },
                        "accuracy_score": accuracy,
                        "trend_direction": trend,
                        "total_forecast": total_forecast
                    }
                    print(f"✅ Receita prevista: R$ {total_forecast:,.0f}, precisão {accuracy:.1f}%, tendência {trend} em {revenue_time:.3f}s")
                else:
                    raise Exception(revenue_data["error"])
                    
            except Exception as e:
                self._log_prophet_test_error(test_report, e, "revenue_forecast_15d")
                print(f"❌ Erro na previsão de receita: {str(e)}")

            # 4. Teste de Previsão de Volume (15 dias)
            test_report["metadata"]["current_stage"] = "volume_forecast_15d"
            print("\n=== ETAPA 4: TESTE DE PREVISÃO DE VOLUME (15 DIAS) ===")
            
            try:
                start_time = time.time()
                print("📦 Testando previsão de volume para 15 dias...")
                
                volume_result = self._run(
                    data_path=data_file_path,
                    target_column="Quantidade",
                    periods=15,
                    aggregation="daily",
                    seasonality_mode="multiplicative",
                    include_holidays=True
                )
                volume_time = time.time() - start_time
                
                volume_data = json.loads(volume_result)
                if "error" not in volume_data:
                    volume_accuracy = volume_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                    volume_forecast = volume_data["business_insights"]["total_forecast"]
                    
                    test_report["results"]["volume_forecast"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": volume_time,
                            "periods_forecasted": 15
                        },
                        "accuracy_score": volume_accuracy,
                        "total_forecast": volume_forecast
                    }
                    print(f"✅ Volume previsto: {volume_forecast:,.0f} unidades, precisão {volume_accuracy:.1f}% em {volume_time:.3f}s")
                else:
                    raise Exception(volume_data["error"])
                    
            except Exception as e:
                self._log_prophet_test_error(test_report, e, "volume_forecast_15d")
                print(f"❌ Erro na previsão de volume: {str(e)}")

            # 5. Teste de Diferentes Agregações
            test_report["metadata"]["current_stage"] = "aggregation_testing"
            print("\n=== ETAPA 5: TESTE DE AGREGAÇÕES TEMPORAIS ===")
            
            aggregation_results = {}
            for agg_type in ["daily", "weekly", "monthly"]:
                try:
                    start_time = time.time()
                    print(f"📅 Testando agregação {agg_type}...")
                    
                    agg_result = self._run(
                        data_path=data_file_path,
                        target_column="Total_Liquido",
                        periods=15,
                        aggregation=agg_type,
                        seasonality_mode="multiplicative"
                    )
                    agg_time = time.time() - start_time
                    
                    agg_data = json.loads(agg_result)
                    if "error" not in agg_data:
                        accuracy = agg_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                        aggregation_results[agg_type] = accuracy
                        
                        test_report["results"][f"{agg_type}_aggregation"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": agg_time
                            },
                            "accuracy_score": accuracy
                        }
                        print(f"✅ Agregação {agg_type}: {accuracy:.1f}% precisão em {agg_time:.3f}s")
                    else:
                        raise Exception(agg_data["error"])
                        
                except Exception as e:
                    self._log_prophet_test_error(test_report, e, f"{agg_type}_aggregation")
                    print(f"❌ Erro na agregação {agg_type}: {str(e)}")
            
            # Determinar melhor agregação
            if aggregation_results:
                best_agg = max(aggregation_results, key=aggregation_results.get)
                test_report["results"]["aggregation_analysis"] = {
                    "best_performing": best_agg,
                    "accuracy_scores": aggregation_results
                }

            # 6. Teste de Modos de Sazonalidade
            test_report["metadata"]["current_stage"] = "seasonality_testing"
            print("\n=== ETAPA 6: TESTE DE MODOS DE SAZONALIDADE ===")
            
            seasonality_results = {}
            for season_mode in ["multiplicative", "additive"]:
                try:
                    start_time = time.time()
                    print(f"🌊 Testando sazonalidade {season_mode}...")
                    
                    season_result = self._run(
                        data_path=data_file_path,
                        target_column="Total_Liquido",
                        periods=15,
                        seasonality_mode=season_mode,
                        include_holidays=True
                    )
                    season_time = time.time() - start_time
                    
                    season_data = json.loads(season_result)
                    if "error" not in season_data:
                        accuracy = season_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                        seasonality_results[season_mode] = accuracy
                        
                        test_report["results"][f"{season_mode}_seasonality"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": season_time
                            },
                            "accuracy_score": accuracy
                        }
                        print(f"✅ Sazonalidade {season_mode}: {accuracy:.1f}% precisão em {season_time:.3f}s")
                    else:
                        raise Exception(season_data["error"])
                        
                except Exception as e:
                    self._log_prophet_test_error(test_report, e, f"{season_mode}_seasonality")
                    print(f"❌ Erro na sazonalidade {season_mode}: {str(e)}")
            
            # Análise de sazonalidade
            if seasonality_results:
                best_seasonality = max(seasonality_results, key=seasonality_results.get)
                test_report["results"]["seasonality_analysis"] = {
                    "seasonal_strength": best_seasonality,
                    "accuracy_comparison": seasonality_results
                }

            # 7. Teste de Impacto de Feriados
            test_report["metadata"]["current_stage"] = "holidays_testing"
            print("\n=== ETAPA 7: TESTE DE IMPACTO DE FERIADOS ===")
            
            holidays_results = {}
            for include_holidays in [True, False]:
                try:
                    start_time = time.time()
                    holiday_label = "com" if include_holidays else "sem"
                    print(f"🎄 Testando {holiday_label} feriados...")
                    
                    holiday_result = self._run(
                        data_path=data_file_path,
                        target_column="Total_Liquido",
                        periods=15,
                        include_holidays=include_holidays,
                        seasonality_mode="multiplicative"
                    )
                    holiday_time = time.time() - start_time
                    
                    holiday_data = json.loads(holiday_result)
                    if "error" not in holiday_data:
                        accuracy = holiday_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                        holidays_results[holiday_label] = accuracy
                        
                        test_report["results"][f"{'with' if include_holidays else 'without'}_holidays"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": holiday_time
                            },
                            "accuracy_score": accuracy
                        }
                        print(f"✅ {holiday_label.capitalize()} feriados: {accuracy:.1f}% precisão em {holiday_time:.3f}s")
                    else:
                        raise Exception(holiday_data["error"])
                        
                except Exception as e:
                    self._log_prophet_test_error(test_report, e, f"{'with' if include_holidays else 'without'}_holidays")
                    print(f"❌ Erro {holiday_label} feriados: {str(e)}")

            # 8. Teste de Diferentes Períodos
            test_report["metadata"]["current_stage"] = "periods_testing"
            print("\n=== ETAPA 8: TESTE DE DIFERENTES PERÍODOS ===")
            
            config_tests = {}
            for periods in [7, 15, 30]:
                try:
                    start_time = time.time()
                    print(f"📈 Testando previsão para {periods} dias...")
                    
                    period_result = self._run(
                        data_path=data_file_path,
                        target_column="Total_Liquido",
                        periods=periods,
                        aggregation="daily"
                    )
                    period_time = time.time() - start_time
                    
                    period_data = json.loads(period_result)
                    if "error" not in period_data:
                        accuracy = period_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                        config_tests[f"{periods}_days"] = {
                            "success": True,
                            "description": f"Previsão para {periods} dias",
                            "accuracy": accuracy,
                            "execution_time": period_time
                        }
                        print(f"✅ {periods} dias: {accuracy:.1f}% precisão em {period_time:.3f}s")
                    else:
                        config_tests[f"{periods}_days"] = {
                            "success": False,
                            "error": period_data["error"]
                        }
                        
                except Exception as e:
                    config_tests[f"{periods}_days"] = {"success": False, "error": str(e)}
                    print(f"❌ Erro em {periods} dias: {str(e)}")
            
            test_report["component_tests"]["configuration_tests"] = config_tests

            # 9. Validação de Modelos
            test_report["metadata"]["current_stage"] = "model_validation"
            print("\n=== ETAPA 9: VALIDAÇÃO DE MODELOS ===")
            
            try:
                start_time = time.time()
                print("🔍 Executando validação completa...")
                
                # Teste com melhor configuração encontrada
                best_config_result = self._run(
                    data_path=data_file_path,
                    target_column="Total_Liquido",
                    periods=15,
                    aggregation="daily",
                    seasonality_mode="multiplicative",
                    include_holidays=True,
                    confidence_interval=0.80
                )
                validation_time = time.time() - start_time
                
                validation_data = json.loads(best_config_result)
                if "error" not in validation_data:
                    test_report["results"]["model_validation"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": validation_time
                        },
                        "final_accuracy": validation_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                    }
                    
                    # Extrair insights para componente de testes
                    test_report["component_tests"]["forecast_insights"] = {
                        "trend_direction": validation_data["business_insights"]["trend_direction"],
                        "trend_percentage": validation_data["business_insights"]["trend_percentage"],
                        "total_forecast": validation_data["business_insights"]["total_forecast"],
                        "peak_day_value": validation_data["business_insights"]["peak_day"]["value"],
                        "model_accuracy": validation_data["forecast_summary"]["model_accuracy"]["accuracy_score"]
                    }
                    
                    print(f"✅ Validação concluída: {validation_data['forecast_summary']['model_accuracy']['accuracy_score']:.1f}% precisão em {validation_time:.3f}s")
                else:
                    raise Exception(validation_data["error"])
                    
            except Exception as e:
                self._log_prophet_test_error(test_report, e, "model_validation")
                print(f"❌ Erro na validação: {str(e)}")

            # 10. Contagem de modelos testados
            test_report["data_metrics"]["models_tested"] = len([r for r in test_report["results"].values() if r.get("success")])

            # 11. Performance Metrics
            test_report["performance_metrics"] = {
                "total_execution_time": sum([
                    result.get('metrics', {}).get('processing_time', 0) 
                    for result in test_report["results"].values() 
                    if isinstance(result, dict)
                ]),
                "memory_usage_mb": self._get_prophet_memory_usage(),
                "models_successfully_trained": test_report["data_metrics"]["models_tested"]
            }

            # 12. Análise Final
            test_report["metadata"]["status"] = "completed" if not test_report["errors"] else "completed_with_errors"
            print(f"\n✅✅✅ TESTE PROPHET COMPLETO - {len(test_report['errors'])} erros ✅✅✅")
            
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            test_report["metadata"]["status"] = "failed"
            self._log_prophet_test_error(test_report, e, "global")
            print(f"❌ TESTE PROPHET FALHOU: {str(e)}")
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

    def _log_prophet_test_error(self, report: dict, error: Exception, context: str) -> None:
        """Registra erros de teste Prophet de forma estruturada"""
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        report["errors"].append(error_entry)

    def _perform_prophet_data_quality_check(self, df: pd.DataFrame) -> dict:
        """Executa verificações de qualidade específicas para dados Prophet"""
        checks = {
            "missing_dates": int(df['Data'].isnull().sum()) if 'Data' in df.columns else 0,
            "missing_target_values": int(df['Total_Liquido'].isnull().sum()) if 'Total_Liquido' in df.columns else 0,
            "negative_values": int((df['Total_Liquido'] < 0).sum()) if 'Total_Liquido' in df.columns else 0,
            "zero_values": int((df['Total_Liquido'] == 0).sum()) if 'Total_Liquido' in df.columns else 0,
            "duplicate_dates": int(df.duplicated(subset=['Data']).sum()) if 'Data' in df.columns else 0,
            "data_gaps": self._detect_date_gaps(df) if 'Data' in df.columns else 0
        }
        return checks

    def _detect_date_gaps(self, df: pd.DataFrame) -> int:
        """Detecta lacunas na série temporal"""
        try:
            df['Data'] = pd.to_datetime(df['Data'])
            date_range = pd.date_range(start=df['Data'].min(), end=df['Data'].max(), freq='D')
            missing_dates = len(date_range) - df['Data'].nunique()
            return missing_dates
        except:
            return 0

    def _get_prophet_memory_usage(self) -> float:
        """Obtém uso de memória específico para análises Prophet"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Em MB
        except:
            return 0.0


# Exemplo de uso
if __name__ == "__main__":
    tool = ProphetForecastTool()
    
    print("🔮 Iniciando Teste Completo do Sistema Prophet...")
    print("📁 Testando especificamente com: data/vendas.csv")
    
    # Executar teste usando especificamente data/vendas.csv
    report = tool.run_full_prophet_test()
    
    # Salvar relatório
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/prophet_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Relatório Prophet gerado em test_results/prophet_test_report.md")
    print(f"📁 Teste executado com arquivo: data/vendas.csv")
    print("\n" + "="*80)
    print(report[:1500])  # Exibir parte do relatório no console