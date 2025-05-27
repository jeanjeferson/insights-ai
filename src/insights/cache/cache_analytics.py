#!/usr/bin/env python
"""
ETAPA 4 - ANALYTICS DE CACHE
Sistema de analytics e relatórios para o cache inteligente
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from .intelligent_cache import IntelligentCacheSystem, CacheEntry, CacheType

# Configuração de logging
analytics_logger = logging.getLogger('cache_analytics')
analytics_logger.setLevel(logging.INFO)

@dataclass
class CacheMetric:
    """Métrica individual do cache"""
    timestamp: datetime
    metric_name: str
    value: float
    metadata: Dict[str, Any]

@dataclass
class CacheReport:
    """Relatório de analytics do cache"""
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    summary_stats: Dict[str, Any]
    detailed_metrics: List[CacheMetric]
    recommendations: List[str]

class CacheAnalytics:
    """Sistema de analytics para cache inteligente"""
    
    def __init__(self, cache_system: IntelligentCacheSystem, 
                 metrics_retention_days: int = 7):
        self.cache_system = cache_system
        self.metrics_retention_days = metrics_retention_days
        
        # Histórico de métricas
        self.metrics_history: List[CacheMetric] = []
        
        # Configuração de métricas
        self.tracked_metrics = [
            "hit_rate",
            "miss_rate", 
            "memory_usage",
            "disk_usage",
            "eviction_rate",
            "cache_efficiency",
            "response_time"
        ]
        
        analytics_logger.info("📊 CacheAnalytics inicializado")
    
    def collect_metrics(self) -> Dict[str, float]:
        """Coletar métricas atuais do cache"""
        try:
            current_stats = self.cache_system.get_stats()
            now = datetime.now()
            
            # Calcular métricas derivadas
            metrics = {}
            
            # Taxa de hit/miss
            total_requests = current_stats.get("hits", 0) + current_stats.get("misses", 0)
            if total_requests > 0:
                metrics["hit_rate"] = current_stats.get("hits", 0) / total_requests * 100
                metrics["miss_rate"] = current_stats.get("misses", 0) / total_requests * 100
            else:
                metrics["hit_rate"] = 0.0
                metrics["miss_rate"] = 0.0
            
            # Uso de memória e disco
            metrics["memory_usage"] = current_stats.get("memory_usage_percent", 0.0)
            metrics["disk_usage"] = current_stats.get("disk_usage_percent", 0.0)
            
            # Taxa de eviction
            metrics["eviction_rate"] = current_stats.get("evictions", 0)
            
            # Eficiência do cache (baseada na combinação de hit rate e uso de recursos)
            hit_rate = metrics["hit_rate"]
            memory_efficiency = 100 - metrics["memory_usage"]
            metrics["cache_efficiency"] = (hit_rate * 0.7 + memory_efficiency * 0.3)
            
            # Tempo de resposta (simulado baseado no estado do cache)
            base_response_time = 10  # ms
            if metrics["memory_usage"] > 80:
                response_penalty = (metrics["memory_usage"] - 80) * 2
            else:
                response_penalty = 0
            metrics["response_time"] = base_response_time + response_penalty
            
            # Armazenar métricas no histórico
            for metric_name, value in metrics.items():
                metric = CacheMetric(
                    timestamp=now,
                    metric_name=metric_name,
                    value=value,
                    metadata={"raw_stats": current_stats}
                )
                self.metrics_history.append(metric)
            
            # Limpar métricas antigas
            self._cleanup_old_metrics()
            
            analytics_logger.debug(f"📈 Métricas coletadas: {len(metrics)} métricas")
            return metrics
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao coletar métricas: {e}")
            return {}
    
    def generate_report(self, 
                       period_hours: int = 24,
                       include_recommendations: bool = True) -> CacheReport:
        """Gerar relatório de analytics"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=period_hours)
            
            # Filtrar métricas do período
            period_metrics = [
                m for m in self.metrics_history
                if start_time <= m.timestamp <= end_time
            ]
            
            # Calcular estatísticas resumidas
            summary_stats = self._calculate_summary_stats(period_metrics)
            
            # Gerar recomendações
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_recommendations(summary_stats, period_metrics)
            
            report = CacheReport(
                generated_at=end_time,
                period_start=start_time,
                period_end=end_time,
                summary_stats=summary_stats,
                detailed_metrics=period_metrics,
                recommendations=recommendations
            )
            
            analytics_logger.info(f"📋 Relatório gerado: {len(period_metrics)} métricas, {period_hours}h período")
            return report
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao gerar relatório: {e}")
            return CacheReport(
                generated_at=datetime.now(),
                period_start=datetime.now(),
                period_end=datetime.now(),
                summary_stats={},
                detailed_metrics=[],
                recommendations=["Erro ao gerar relatório"]
            )
    
    def export_report(self, report: CacheReport, 
                     output_dir: str = "data/cache/reports") -> str:
        """Exportar relatório para arquivo JSON"""
        try:
            # Criar diretório se não existir
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Nome do arquivo baseado no timestamp
            filename = f"cache_report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = output_path / filename
            
            # Converter relatório para dicionário serializável
            report_data = {
                "generated_at": report.generated_at.isoformat(),
                "period_start": report.period_start.isoformat(),
                "period_end": report.period_end.isoformat(),
                "summary_stats": report.summary_stats,
                "detailed_metrics": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "metric_name": m.metric_name,
                        "value": m.value,
                        "metadata": m.metadata
                    }
                    for m in report.detailed_metrics
                ],
                "recommendations": report.recommendations
            }
            
            # Salvar arquivo
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            analytics_logger.info(f"💾 Relatório exportado: {filepath}")
            return str(filepath)
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao exportar relatório: {e}")
            return ""
    
    def get_metric_trend(self, metric_name: str, 
                        hours_back: int = 24) -> List[Dict[str, Any]]:
        """Obter tendência de uma métrica específica"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filtrar métricas
            filtered_metrics = [
                m for m in self.metrics_history
                if m.metric_name == metric_name and m.timestamp >= cutoff_time
            ]
            
            # Converter para formato de série temporal
            trend_data = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "value": m.value,
                    "hour": m.timestamp.hour,
                    "day": m.timestamp.weekday()
                }
                for m in sorted(filtered_metrics, key=lambda x: x.timestamp)
            ]
            
            return trend_data
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao obter tendência de {metric_name}: {e}")
            return []
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Obter insights de performance do cache"""
        try:
            current_stats = self.cache_system.get_stats()
            recent_metrics = self.get_metric_trend("hit_rate", 1)  # Última hora
            
            insights = {
                "overall_health": "unknown",
                "hit_rate_trend": "stable",
                "memory_pressure": "normal",
                "disk_pressure": "normal",
                "optimization_score": 0.0,
                "key_findings": []
            }
            
            # Avaliar saúde geral
            hit_rate = current_stats.get("hit_rate_percent", 0)
            memory_usage = current_stats.get("memory_usage_percent", 0)
            disk_usage = current_stats.get("disk_usage_percent", 0)
            
            if hit_rate > 70 and memory_usage < 80 and disk_usage < 80:
                insights["overall_health"] = "excellent"
            elif hit_rate > 50 and memory_usage < 90 and disk_usage < 90:
                insights["overall_health"] = "good"
            elif hit_rate > 30:
                insights["overall_health"] = "fair"
            else:
                insights["overall_health"] = "poor"
            
            # Avaliar tendência de hit rate
            if len(recent_metrics) >= 2:
                first_value = recent_metrics[0]["value"]
                last_value = recent_metrics[-1]["value"]
                change = last_value - first_value
                
                if change > 5:
                    insights["hit_rate_trend"] = "improving"
                elif change < -5:
                    insights["hit_rate_trend"] = "declining"
                else:
                    insights["hit_rate_trend"] = "stable"
            
            # Avaliar pressão de memória
            if memory_usage > 90:
                insights["memory_pressure"] = "critical"
            elif memory_usage > 80:
                insights["memory_pressure"] = "high"
            elif memory_usage > 60:
                insights["memory_pressure"] = "moderate"
            else:
                insights["memory_pressure"] = "normal"
            
            # Avaliar pressão de disco
            if disk_usage > 90:
                insights["disk_pressure"] = "critical"
            elif disk_usage > 80:
                insights["disk_pressure"] = "high"
            elif disk_usage > 60:
                insights["disk_pressure"] = "moderate"
            else:
                insights["disk_pressure"] = "normal"
            
            # Calcular score de otimização
            optimization_factors = [
                hit_rate / 100.0,  # Hit rate normalizado
                (100 - memory_usage) / 100.0,  # Eficiência de memória
                (100 - disk_usage) / 100.0,  # Eficiência de disco
                min(current_stats.get("total_entries", 0) / 1000.0, 1.0)  # Utilização
            ]
            insights["optimization_score"] = sum(optimization_factors) / len(optimization_factors) * 100
            
            # Gerar achados principais
            findings = []
            if hit_rate < 50:
                findings.append("Hit rate baixo - considere ajustar estratégia de cache")
            if memory_usage > 85:
                findings.append("Uso de memória alto - considere aumentar limite ou eviction")
            if disk_usage > 85:
                findings.append("Uso de disco alto - considere limpeza ou aumento de limite")
            if current_stats.get("evictions", 0) > 100:
                findings.append("Muitas evictions - considere aumentar capacidade")
            
            insights["key_findings"] = findings
            
            return insights
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao gerar insights: {e}")
            return {"error": str(e)}
    
    def _calculate_summary_stats(self, metrics: List[CacheMetric]) -> Dict[str, Any]:
        """Calcular estatísticas resumidas"""
        try:
            if not metrics:
                return {}
            
            # Agrupar métricas por nome
            metrics_by_name = {}
            for metric in metrics:
                if metric.metric_name not in metrics_by_name:
                    metrics_by_name[metric.metric_name] = []
                metrics_by_name[metric.metric_name].append(metric.value)
            
            # Calcular estatísticas para cada métrica
            summary = {}
            for metric_name, values in metrics_by_name.items():
                if values:
                    summary[metric_name] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "current": values[-1] if values else 0,
                        "count": len(values)
                    }
            
            # Estatísticas gerais
            summary["_general"] = {
                "total_metrics": len(metrics),
                "unique_metric_types": len(metrics_by_name),
                "collection_period_hours": (
                    (metrics[-1].timestamp - metrics[0].timestamp).total_seconds() / 3600
                    if len(metrics) > 1 else 0
                )
            }
            
            return summary
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao calcular estatísticas: {e}")
            return {}
    
    def _generate_recommendations(self, 
                                 summary_stats: Dict[str, Any],
                                 metrics: List[CacheMetric]) -> List[str]:
        """Gerar recomendações baseadas nas métricas"""
        recommendations = []
        
        try:
            # Recomendações baseadas em hit rate
            hit_rate_stats = summary_stats.get("hit_rate", {})
            avg_hit_rate = hit_rate_stats.get("avg", 0)
            
            if avg_hit_rate < 30:
                recommendations.append(
                    "🔴 Hit rate muito baixo (<30%). Considere revisar estratégia de cache e TTL."
                )
            elif avg_hit_rate < 60:
                recommendations.append(
                    "🟡 Hit rate moderado (<60%). Considere otimizar condições de cache."
                )
            else:
                recommendations.append(
                    "🟢 Hit rate bom (>60%). Cache funcionando eficientemente."
                )
            
            # Recomendações baseadas em uso de memória
            memory_stats = summary_stats.get("memory_usage", {})
            avg_memory = memory_stats.get("avg", 0)
            max_memory = memory_stats.get("max", 0)
            
            if max_memory > 95:
                recommendations.append(
                    "🔴 Uso de memória crítico (>95%). Aumente capacidade ou melhore eviction."
                )
            elif avg_memory > 80:
                recommendations.append(
                    "🟡 Uso de memória alto (>80%). Monitore crescimento e considere otimização."
                )
            
            # Recomendações baseadas em evictions
            eviction_stats = summary_stats.get("eviction_rate", {})
            avg_evictions = eviction_stats.get("avg", 0)
            
            if avg_evictions > 50:
                recommendations.append(
                    "🟡 Taxa de eviction alta. Considere aumentar capacidade ou ajustar TTL."
                )
            
            # Recomendações baseadas em eficiência
            efficiency_stats = summary_stats.get("cache_efficiency", {})
            avg_efficiency = efficiency_stats.get("avg", 0)
            
            if avg_efficiency < 50:
                recommendations.append(
                    "🔴 Eficiência do cache baixa (<50%). Revise configurações gerais."
                )
            elif avg_efficiency > 80:
                recommendations.append(
                    "🟢 Eficiência do cache excelente (>80%). Configuração otimizada."
                )
            
            # Recomendações temporais
            if len(metrics) > 0:
                recent_metrics = [m for m in metrics if m.timestamp > datetime.now() - timedelta(hours=1)]
                if len(recent_metrics) < 5:
                    recommendations.append(
                        "ℹ️ Poucos dados recentes. Considere aumentar frequência de coleta de métricas."
                    )
            
            return recommendations
            
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao gerar recomendações: {e}")
            return ["Erro ao gerar recomendações"]
    
    def _cleanup_old_metrics(self):
        """Limpar métricas antigas baseado na retenção configurada"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
            original_count = len(self.metrics_history)
            
            self.metrics_history = [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
            ]
            
            removed_count = original_count - len(self.metrics_history)
            if removed_count > 0:
                analytics_logger.debug(f"🧹 Métricas antigas removidas: {removed_count}")
                
        except Exception as e:
            analytics_logger.error(f"❌ Erro ao limpar métricas antigas: {e}")

# ========== FUNÇÕES UTILITÁRIAS ==========

def create_analytics_dashboard_data(analytics: CacheAnalytics) -> Dict[str, Any]:
    """Criar dados para dashboard de analytics"""
    try:
        # Coletar métricas atuais
        current_metrics = analytics.collect_metrics()
        
        # Obter insights
        insights = analytics.get_performance_insights()
        
        # Obter tendências das últimas 24h
        trends = {}
        for metric in ["hit_rate", "memory_usage", "cache_efficiency"]:
            trends[metric] = analytics.get_metric_trend(metric, 24)
        
        dashboard_data = {
            "current_metrics": current_metrics,
            "performance_insights": insights,
            "trends_24h": trends,
            "last_updated": datetime.now().isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        analytics_logger.error(f"❌ Erro ao criar dados do dashboard: {e}")
        return {"error": str(e)} 