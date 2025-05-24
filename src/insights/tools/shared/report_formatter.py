"""
📊 MÓDULO DE FORMATAÇÃO DE RELATÓRIOS CONSOLIDADO
=================================================

Este módulo centraliza toda a lógica de formatação de relatórios,
eliminando duplicação entre KPI Calculator Tool e Statistical Analysis Tool.

FUNCIONALIDADES:
✅ Formatação de relatórios de KPIs de negócio
✅ Formatação de relatórios de análises estatísticas
✅ Formatação de insights e alertas
✅ Formatação de dados tabulares
✅ Tratamento de erros na formatação
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from datetime import datetime
import json

class ReportFormatterMixin:
    """Mixin class para formatação de relatórios de joalherias."""
    
    def format_business_kpi_report(self, kpis: Dict[str, Any], categoria: str = "all", 
                                  benchmark_mode: bool = True) -> str:
        """
        Formatar relatório de KPIs de negócio de forma padronizada.
        
        Args:
            kpis: Dicionário com KPIs calculados
            categoria: Categoria analisada
            benchmark_mode: Se inclui benchmarks
            
        Returns:
            Relatório formatado como string
        """
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            report = f"""
            # 📊 RELATÓRIO DE KPIs EMPRESARIAIS - JOALHERIA
            **Versão**: 3.0 - Otimizada e Consolidada
            **Categoria**: {categoria.upper()} | **Data**: {timestamp}
            {'**Inclui Benchmarks do Setor**' if benchmark_mode else ''}

            ## 🚨 ALERTAS AUTOMÁTICOS
            """
            
            # Adicionar alertas
            if 'alertas' in kpis and kpis['alertas']:
                for alert in kpis['alertas']:
                    report += f"{alert}\n"
            else:
                report += "✅ Nenhum alerta crítico identificado\n"
            
            report += "\n## 💡 INSIGHTS PRINCIPAIS\n"
            
            # Adicionar insights
            if 'insights' in kpis and kpis['insights']:
                for i, insight in enumerate(kpis['insights'], 1):
                    report += f"{i}. {insight}\n"
            else:
                report += "📊 Análise em andamento - insights serão gerados conforme dados disponíveis\n"
            
            report += "\n---\n"
            
            # Formatar seções de KPIs
            for section_name, section_data in kpis.items():
                if section_name in ['insights', 'alertas']:
                    continue
                
                if isinstance(section_data, dict) and not any(key in section_data for key in ['erro', 'error']):
                    section_title = self._get_kpi_section_title(section_name)
                    report += f"\n## {section_title}\n\n"
                    report += self._format_kpi_section_data(section_data, section_name)
                elif isinstance(section_data, dict) and ('erro' in section_data or 'error' in section_data):
                    section_title = self._get_kpi_section_title(section_name)
                    report += f"\n## {section_title}\n\n"
                    error_msg = section_data.get('erro', section_data.get('error', 'Erro desconhecido'))
                    report += f"❌ {error_msg}\n\n"
            
            # Rodapé
            report += self._get_kpi_report_footer()
            
            return report
            
        except Exception as e:
            return f"Erro na formatação do relatório de KPIs: {str(e)}"
    
    def format_statistical_analysis_report(self, result: Dict[str, Any], analysis_type: str) -> str:
        """
        Formatar relatório de análise estatística de forma padronizada.
        
        Args:
            result: Resultado da análise estatística
            analysis_type: Tipo de análise realizada
            
        Returns:
            Relatório formatado como string
        """
        try:
            if 'error' in result:
                return f"❌ Erro na análise {analysis_type}: {result['error']}"
            
            # Header específico para análises estatísticas
            formatted = f"""
                        ✨ ANÁLISE ESTATÍSTICA AVANÇADA: {analysis_type.upper().replace('_', ' ')}
                        ═══════════════════════════════════════════════════════════════════

                        🎯 ANÁLISE ESPECIALIZADA PARA JOALHERIAS
                        📊 Dados Demográficos ✅ | Dados Geográficos ✅ | Dados Financeiros ✅

                        """
            
            # Formatação específica por tipo de análise
            if analysis_type in ['demographic_patterns', 'geographic_performance', 'profitability_analysis']:
                formatted += self._format_business_specific_analysis(result, analysis_type)
            else:
                formatted += self._format_standard_statistical_analysis(result)
            
            # Insights em destaque
            if 'insights' in result and result['insights']:
                formatted += f"\n🎯 INSIGHTS ESTATÍSTICOS:\n"
                for insight in result['insights']:
                    formatted += f"   • {insight}\n"
            
            # Footer estatístico
            formatted += self._get_statistical_report_footer()
            
            return formatted
            
        except Exception as e:
            return f"❌ Erro na formatação da análise estatística: {str(e)}"
    
    def format_data_table(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                         title: str = "", max_rows: int = 20) -> str:
        """
        Formatar dados tabulares de forma consistente.
        
        Args:
            data: DataFrame ou dicionário com dados
            title: Título da tabela
            max_rows: Máximo de linhas a exibir
            
        Returns:
            Tabela formatada como string
        """
        try:
            formatted = f"\n### {title}\n" if title else "\n"
            
            if isinstance(data, pd.DataFrame):
                if len(data) > max_rows:
                    formatted += f"**Mostrando {max_rows} de {len(data)} registros**\n\n"
                    data_to_show = data.head(max_rows)
                else:
                    data_to_show = data
                
                # Formatar DataFrame como markdown table
                formatted += data_to_show.to_markdown(index=True, floatfmt=".2f")
                
            elif isinstance(data, dict):
                formatted += self._format_dict_as_table(data, max_items=max_rows)
            
            return formatted + "\n"
            
        except Exception as e:
            return f"❌ Erro na formatação da tabela: {str(e)}\n"
    
    def format_insights_section(self, insights: List[str], title: str = "💡 INSIGHTS") -> str:
        """
        Formatar seção de insights de forma padronizada.
        
        Args:
            insights: Lista de insights
            title: Título da seção
            
        Returns:
            Seção de insights formatada
        """
        if not insights:
            return f"\n## {title}\n📊 Nenhum insight específico identificado nesta análise.\n"
        
        formatted = f"\n## {title}\n"
        for i, insight in enumerate(insights, 1):
            formatted += f"{i}. {insight}\n"
        
        return formatted + "\n"
    
    def format_alerts_section(self, alerts: List[str], title: str = "🚨 ALERTAS") -> str:
        """
        Formatar seção de alertas de forma padronizada.
        
        Args:
            alerts: Lista de alertas
            title: Título da seção
            
        Returns:
            Seção de alertas formatada
        """
        if not alerts:
            return f"\n## {title}\n✅ Nenhum alerta crítico identificado.\n"
        
        formatted = f"\n## {title}\n"
        for alert in alerts:
            formatted += f"{alert}\n"
        
        return formatted + "\n"
    
    def format_numeric_value(self, value: Union[int, float], value_type: str = "number") -> str:
        """
        Formatar valores numéricos de acordo com o tipo.
        
        Args:
            value: Valor numérico
            value_type: Tipo do valor ('currency', 'percentage', 'number', 'count')
            
        Returns:
            Valor formatado como string
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            if value_type == "currency":
                return f"R$ {value:,.2f}"
            elif value_type == "percentage":
                return f"{value:.2f}%"
            elif value_type == "count":
                return f"{int(value):,}"
            else:  # number
                return f"{value:,.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    def _get_kpi_section_title(self, section_name: str) -> str:
        """Obter título formatado para seção de KPIs."""
        titles = {
            'financeiros': '💰 KPIs FINANCEIROS',
            'operacionais': '⚙️ KPIs OPERACIONAIS', 
            'inventario': '📦 KPIs DE INVENTÁRIO',
            'clientes': '👥 KPIs DE CLIENTES',
            'equipe_vendas': '👨‍💼 KPIs DA EQUIPE DE VENDAS',
            'geograficos': '🗺️ KPIs GEOGRÁFICOS',
            'produtos': '💎 KPIs DE PRODUTOS',
            'benchmarks': '📈 COMPARAÇÃO COM BENCHMARKS'
        }
        return titles.get(section_name, section_name.upper().replace('_', ' '))
    
    def _format_kpi_section_data(self, data: Dict[str, Any], section_name: str) -> str:
        """Formatar dados de uma seção específica de KPIs."""
        formatted = ""
        
        try:
            for key, value in data.items():
                if key in ['erro', 'error']:
                    formatted += f"❌ {value}\n\n"
                    continue
                    
                if isinstance(value, dict):
                    formatted += f"### {key.replace('_', ' ').title()}\n"
                    for subkey, subvalue in value.items():
                        formatted += self._format_kpi_value(subkey, subvalue)
                    formatted += "\n"
                elif isinstance(value, list):
                    formatted += f"**{key.replace('_', ' ').title()}**:\n"
                    for item in value[:5]:  # Limitar a 5 itens
                        formatted += f"  - {item}\n"
                    formatted += "\n"
                else:
                    formatted += self._format_kpi_value(key, value)
                    
            return formatted
            
        except Exception as e:
            return f"Erro na formatação da seção {section_name}: {str(e)}\n"
    
    def _format_kpi_value(self, key: str, value) -> str:
        """Formatar valor individual de KPI."""
        if isinstance(value, (int, float)):
            if any(word in key.lower() for word in ['pct', 'rate', 'percentage', 'percentual']):
                return f"- **{key.replace('_', ' ').title()}**: {self.format_numeric_value(value, 'percentage')}\n"
            elif any(word in key.lower() for word in ['revenue', 'total', 'aov', 'margem', 'valor', 'clv', 'receita', 'ticket']):
                return f"- **{key.replace('_', ' ').title()}**: {self.format_numeric_value(value, 'currency')}\n"
            elif any(word in key.lower() for word in ['count', 'produtos', 'clientes', 'vendedores']):
                return f"- **{key.replace('_', ' ').title()}**: {self.format_numeric_value(value, 'count')}\n"
            else:
                return f"- **{key.replace('_', ' ').title()}**: {self.format_numeric_value(value, 'number')}\n"
        elif isinstance(value, dict) and len(value) < 10:
            items = [f"{k}: {self.format_numeric_value(v, 'number')}" for k, v in list(value.items())[:3]]
            return f"- **{key.replace('_', ' ').title()}**: {', '.join(items)}{'...' if len(value) > 3 else ''}\n"
        else:
            return f"- **{key.replace('_', ' ').title()}**: {value}\n"
    
    def _format_business_specific_analysis(self, result: Dict[str, Any], analysis_type: str) -> str:
        """Formatação específica para análises de negócio."""
        formatted = ""
        
        if analysis_type == 'demographic_patterns':
            formatted += "👥 PADRÕES DEMOGRÁFICOS IDENTIFICADOS:\n"
            if 'age_patterns' in result:
                formatted += "   📊 Análise por Faixa Etária:\n"
                formatted += self._format_nested_dict(result['age_patterns'], indent=2)
        
        elif analysis_type == 'geographic_performance':
            formatted += "🗺️ PERFORMANCE GEOGRÁFICA:\n"
            if 'state_performance' in result:
                formatted += "   📍 Análise por Estado:\n"
                formatted += self._format_nested_dict(result['state_performance'], indent=2)
        
        elif analysis_type == 'profitability_analysis':
            formatted += "💰 ANÁLISE DE RENTABILIDADE:\n"
            if 'margin_analysis' in result:
                margin_data = result['margin_analysis']
                formatted += f"   💵 Margem Média: {self.format_numeric_value(margin_data.get('margem_media', 0), 'currency')}\n"
                formatted += f"   📊 Margem Total: {self.format_numeric_value(margin_data.get('margem_total', 0), 'currency')}\n"
        
        return formatted
    
    def _format_standard_statistical_analysis(self, result: Dict[str, Any]) -> str:
        """Formatação padrão para análises estatísticas."""
        formatted = ""
        
        for key, value in result.items():
            if key == 'insights':
                continue  # Tratado separadamente
            
            section_title = key.upper().replace('_', ' ')
            formatted += f"📊 {section_title}:\n"
            
            if isinstance(value, dict):
                formatted += self._format_nested_dict(value, level=1)
            elif isinstance(value, (int, float)):
                formatted += f"   {self.format_numeric_value(value, 'number')}\n"
            else:
                formatted += f"   {value}\n"
            
            formatted += "\n"
        
        return formatted
    
    def _format_nested_dict(self, data: Dict, level: int = 0, indent: int = 1) -> str:
        """Formatar dicionário aninhado recursivamente."""
        formatted = ""
        base_indent = "   " * (level + indent)
        
        for key, value in data.items():
            if isinstance(value, dict):
                formatted += f"{base_indent}🔸 {key.replace('_', ' ').title()}:\n"
                formatted += self._format_nested_dict(value, level + 1, indent)
            elif isinstance(value, (int, float)):
                # Detectar tipo de valor baseado na chave
                if any(word in key.lower() for word in ['pct', 'percentage', 'rate']):
                    formatted += f"{base_indent}• {key.replace('_', ' ').title()}: {self.format_numeric_value(value, 'percentage')}\n"
                elif any(word in key.lower() for word in ['revenue', 'total', 'valor', 'receita']):
                    formatted += f"{base_indent}• {key.replace('_', ' ').title()}: {self.format_numeric_value(value, 'currency')}\n"
                else:
                    formatted += f"{base_indent}• {key.replace('_', ' ').title()}: {self.format_numeric_value(value, 'number')}\n"
            elif isinstance(value, list):
                formatted += f"{base_indent}• {key.replace('_', ' ').title()}: {len(value)} itens\n"
            else:
                formatted += f"{base_indent}• {key.replace('_', ' ').title()}: {value}\n"
        
        return formatted
    
    def _format_dict_as_table(self, data: Dict, max_items: int = 20) -> str:
        """Formatar dicionário como tabela markdown."""
        items = list(data.items())[:max_items]
        
        if not items:
            return "Nenhum dado disponível"
        
        # Cabeçalho
        formatted = "| Chave | Valor |\n|-------|-------|\n"
        
        # Linhas
        for key, value in items:
            key_formatted = str(key).replace('_', ' ').title()
            if isinstance(value, (int, float)):
                value_formatted = self.format_numeric_value(value, 'number')
            else:
                value_formatted = str(value)[:50]  # Limitar tamanho
            
            formatted += f"| {key_formatted} | {value_formatted} |\n"
        
        return formatted
    
    def _get_kpi_report_footer(self) -> str:
        """Rodapé para relatórios de KPIs."""
        return f"""
                ---
                ## 📋 METODOLOGIA

                **KPIs Calculados**: 50+ métricas especializadas para joalherias
                **Dados Utilizados**: Campos financeiros, demográficos, geográficos e operacionais
                **Benchmarks**: Baseados em estudos do setor de varejo de luxo
                **Alertas**: Sistema inteligente de alertas por threshold

                *Relatório gerado automaticamente pelo Sistema de BI Consolidado - Insights AI v3.0*
                """
    
    def _get_statistical_report_footer(self) -> str:
        """Rodapé para relatórios estatísticos."""
        return f"""
                {'═'*70}
                🕒 Análise realizada em: {datetime.now().strftime('%d/%m/%Y %H:%M')}
                🔧 Ferramenta: Statistical Analysis Tool ENHANCED
                📊 Módulo: Formatação Consolidada v3.0
                """ 