#!/usr/bin/env python
"""
📊 PERFORMANCE ANALYTICS - ETAPA 4
Sistema avançado de análise de performance com insights inteligentes
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import statistics
from pathlib import Path

# Configuração de logging
perf_analytics_logger = logging.getLogger('performance_analytics')
perf_analytics_logger.setLevel(logging.INFO)

class PerformanceMetricType(Enum):
    """Tipos de métricas de performance"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_EFFICIENCY = "resource_efficiency"

class AnalysisTimeframe(Enum):
    """Timeframes para análise"""
    LAST_HOUR = "last_hour"
    LAST_24H = "last_24h"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    CUSTOM = "custom"

@dataclass
class PerformanceMetric:
    """Métrica de performance individual"""
    metric_type: PerformanceMetricType
    value: float
    timestamp: datetime
    flow_id: str
    operation_name: str
    context: Dict[str, Any]

@dataclass
class PerformanceTrend:
    """Tendência de performance"""
    metric_type: PerformanceMetricType
    timeframe: AnalysisTimeframe
    trend_direction: str  # improving, degrading, stable
    change_percent: float
    current_value: float
    previous_value: float
    confidence: float
    data_points: int

@dataclass
class PerformanceInsight:
    """Insight de performance"""
    insight_type: str
    title: str
    description: str
    severity: str  # info, warning, critical
    confidence: float
    impact_score: float  # 0-100
    recommendations: List[str]
    supporting_data: Dict[str, Any]

@dataclass
class PerformanceBenchmark:
    """Benchmark de performance"""
    benchmark_name: str
    baseline_value: float
    current_value: float
    improvement_percent: float
    last_updated: datetime
    target_value: Optional[float] = None

class PerformanceAnalytics:
    """
    📊 Sistema de Análise de Performance
    
    Fornece análise avançada de performance com:
    - Coleta e análise de métricas
    - Detecção de tendências
    - Insights automáticos
    - Benchmarking
    - Alertas proativos
    """
    
    def __init__(self, 
                 analytics_dir: str = "data/optimization/analytics",
                 retention_days: int = 30,
                 enable_real_time_analysis: bool = True,
                 insight_generation_interval: int = 300):  # 5 minutos
        
        self.analytics_dir = Path(analytics_dir)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações
        self.retention_days = retention_days
        self.enable_real_time_analysis = enable_real_time_analysis
        self.insight_generation_interval = insight_generation_interval
        
        # Armazenamento de dados
        self.metrics_buffer: List[PerformanceMetric] = []
        self.historical_metrics: Dict[str, List[PerformanceMetric]] = {}
        self.trends: Dict[str, PerformanceTrend] = {}
        self.insights: List[PerformanceInsight] = []
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Estado interno
        self.last_insight_generation = datetime.now()
        self.last_trend_analysis = datetime.now()
        
        # Configurações de análise
        self.analysis_config = {
            "trend_detection_window": 24,  # horas
            "minimum_data_points": 5,
            "outlier_threshold": 2.0,  # desvios padrão
            "significant_change_threshold": 10.0  # percentual
        }
        
        # Inicializar sistema
        self._initialize_analytics_system()
        
        perf_analytics_logger.info("📊 PerformanceAnalytics inicializado")
    
    def _initialize_analytics_system(self):
        """Inicializar sistema de analytics"""
        try:
            # Carregar dados históricos
            self._load_historical_data()
            
            # Inicializar benchmarks padrão
            self._initialize_default_benchmarks()
            
            # Iniciar análise em tempo real se habilitada
            if self.enable_real_time_analysis:
                self._start_real_time_analysis()
            
            perf_analytics_logger.info("✅ Sistema de analytics inicializado")
            
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao inicializar analytics: {e}")
    
    def _load_historical_data(self):
        """Carregar dados históricos"""
        try:
            data_file = self.analytics_dir / "historical_metrics.json"
            if data_file.exists():
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Processar dados carregados (implementação simplificada)
                    perf_analytics_logger.info(f"📁 Dados históricos carregados: {len(data)} registros")
        except Exception as e:
            perf_analytics_logger.warning(f"⚠️ Erro ao carregar dados históricos: {e}")
    
    def _initialize_default_benchmarks(self):
        """Inicializar benchmarks padrão"""
        default_benchmarks = {
            "flow_execution_time": PerformanceBenchmark(
                benchmark_name="Tempo de Execução de Flow",
                baseline_value=60.0,  # 60 segundos
                current_value=60.0,
                improvement_percent=0.0,
                last_updated=datetime.now(),
                target_value=30.0  # Meta: 30 segundos
            ),
            "cache_hit_rate": PerformanceBenchmark(
                benchmark_name="Taxa de Cache Hit",
                baseline_value=50.0,  # 50%
                current_value=50.0,
                improvement_percent=0.0,
                last_updated=datetime.now(),
                target_value=85.0  # Meta: 85%
            ),
            "memory_efficiency": PerformanceBenchmark(
                benchmark_name="Eficiência de Memória",
                baseline_value=512.0,  # 512MB
                current_value=512.0,
                improvement_percent=0.0,
                last_updated=datetime.now(),
                target_value=256.0  # Meta: 256MB
            )
        }
        
        with self.lock:
            self.benchmarks.update(default_benchmarks)
    
    def _start_real_time_analysis(self):
        """Iniciar análise em tempo real"""
        def analysis_loop():
            while self.enable_real_time_analysis:
                try:
                    # Gerar insights se necessário
                    if self._should_generate_insights():
                        self._generate_insights()
                    
                    # Análise de tendências
                    if self._should_analyze_trends():
                        self._analyze_trends()
                    
                    # Limpeza de dados antigos
                    self._cleanup_old_data()
                    
                except Exception as e:
                    perf_analytics_logger.error(f"❌ Erro na análise em tempo real: {e}")
                
                time.sleep(60)  # Verificar a cada minuto
        
        analysis_thread = threading.Thread(target=analysis_loop, daemon=True, name="PerformanceAnalysis")
        analysis_thread.start()
        perf_analytics_logger.info("🔄 Análise em tempo real iniciada")
    
    def record_metric(self, 
                     metric_type: PerformanceMetricType,
                     value: float,
                     flow_id: str,
                     operation_name: str = "unknown",
                     context: Dict[str, Any] = None):
        """Registrar métrica de performance"""
        try:
            metric = PerformanceMetric(
                metric_type=metric_type,
                value=value,
                timestamp=datetime.now(),
                flow_id=flow_id,
                operation_name=operation_name,
                context=context or {}
            )
            
            with self.lock:
                self.metrics_buffer.append(metric)
                
                # Organizar por tipo para análise rápida
                metric_key = f"{metric_type.value}_{operation_name}"
                if metric_key not in self.historical_metrics:
                    self.historical_metrics[metric_key] = []
                
                self.historical_metrics[metric_key].append(metric)
            
            perf_analytics_logger.debug(
                f"📈 Métrica registrada: {metric_type.value} = {value} para {operation_name}"
            )
            
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao registrar métrica: {e}")
    
    def analyze_performance(self, 
                          timeframe: AnalysisTimeframe = AnalysisTimeframe.LAST_24H,
                          flow_id: Optional[str] = None,
                          operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Analisar performance para um timeframe específico"""
        try:
            # Definir período de análise
            end_time = datetime.now()
            if timeframe == AnalysisTimeframe.LAST_HOUR:
                start_time = end_time - timedelta(hours=1)
            elif timeframe == AnalysisTimeframe.LAST_24H:
                start_time = end_time - timedelta(hours=24)
            elif timeframe == AnalysisTimeframe.LAST_WEEK:
                start_time = end_time - timedelta(days=7)
            elif timeframe == AnalysisTimeframe.LAST_MONTH:
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(hours=24)  # Fallback
            
            # Filtrar métricas
            filtered_metrics = self._filter_metrics(start_time, end_time, flow_id, operation_name)
            
            # Calcular estatísticas
            analysis = {
                "timeframe": timeframe.value,
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "metrics_count": len(filtered_metrics),
                "statistics": self._calculate_statistics(filtered_metrics),
                "trends": self._analyze_trends_for_period(filtered_metrics),
                "performance_score": self._calculate_performance_score(filtered_metrics),
                "insights": self._generate_period_insights(filtered_metrics)
            }
            
            return analysis
            
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro na análise de performance: {e}")
            return {"error": str(e)}
    
    def _filter_metrics(self, 
                       start_time: datetime, 
                       end_time: datetime,
                       flow_id: Optional[str] = None,
                       operation_name: Optional[str] = None) -> List[PerformanceMetric]:
        """Filtrar métricas por critérios"""
        filtered = []
        
        with self.lock:
            for metric_list in self.historical_metrics.values():
                for metric in metric_list:
                    # Filtro de tempo
                    if not (start_time <= metric.timestamp <= end_time):
                        continue
                    
                    # Filtro de flow_id
                    if flow_id and metric.flow_id != flow_id:
                        continue
                    
                    # Filtro de operation_name
                    if operation_name and metric.operation_name != operation_name:
                        continue
                    
                    filtered.append(metric)
        
        return filtered
    
    def _calculate_statistics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calcular estatísticas das métricas"""
        if not metrics:
            return {}
        
        # Agrupar por tipo de métrica
        by_type = {}
        for metric in metrics:
            metric_type = metric.metric_type.value
            if metric_type not in by_type:
                by_type[metric_type] = []
            by_type[metric_type].append(metric.value)
        
        statistics_result = {}
        
        for metric_type, values in by_type.items():
            if values:
                statistics_result[metric_type] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "percentile_95": self._calculate_percentile(values, 95),
                    "percentile_99": self._calculate_percentile(values, 99)
                }
        
        return statistics_result
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calcular percentil"""
        try:
            sorted_values = sorted(values)
            index = int((percentile / 100) * len(sorted_values))
            return sorted_values[min(index, len(sorted_values) - 1)]
        except:
            return 0.0
    
    def _analyze_trends_for_period(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analisar tendências para um período"""
        trends = {}
        
        # Agrupar por tipo e calcular tendência
        by_type = {}
        for metric in metrics:
            metric_type = metric.metric_type.value
            if metric_type not in by_type:
                by_type[metric_type] = []
            by_type[metric_type].append((metric.timestamp, metric.value))
        
        for metric_type, time_values in by_type.items():
            if len(time_values) >= 2:
                # Ordenar por tempo
                time_values.sort(key=lambda x: x[0])
                
                # Dividir em primeira e segunda metade
                mid_point = len(time_values) // 2
                first_half = [v[1] for v in time_values[:mid_point]]
                second_half = [v[1] for v in time_values[mid_point:]]
                
                if first_half and second_half:
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    
                    change_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
                    
                    # Determinar direção da tendência
                    if abs(change_percent) < self.analysis_config["significant_change_threshold"]:
                        direction = "stable"
                    elif change_percent > 0:
                        direction = "increasing"
                    else:
                        direction = "decreasing"
                    
                    trends[metric_type] = {
                        "direction": direction,
                        "change_percent": change_percent,
                        "first_period_avg": first_avg,
                        "second_period_avg": second_avg,
                        "data_points": len(time_values)
                    }
        
        return trends
    
    def _calculate_performance_score(self, metrics: List[PerformanceMetric]) -> float:
        """Calcular score geral de performance (0-100)"""
        if not metrics:
            return 50.0  # Neutro
        
        scores = []
        
        # Agrupar por tipo
        by_type = {}
        for metric in metrics:
            metric_type = metric.metric_type.value
            if metric_type not in by_type:
                by_type[metric_type] = []
            by_type[metric_type].append(metric.value)
        
        # Calcular score para cada tipo
        for metric_type, values in by_type.items():
            if not values:
                continue
            
            avg_value = statistics.mean(values)
            
            # Score baseado no tipo de métrica
            if metric_type == "execution_time":
                # Menor é melhor
                if avg_value <= 30:
                    score = 100
                elif avg_value <= 60:
                    score = 80
                elif avg_value <= 120:
                    score = 60
                else:
                    score = 40
            
            elif metric_type == "cache_hit_rate":
                # Maior é melhor (0-100%)
                score = min(avg_value, 100)
            
            elif metric_type == "memory_usage":
                # Menor é melhor (MB)
                if avg_value <= 256:
                    score = 100
                elif avg_value <= 512:
                    score = 80
                elif avg_value <= 1024:
                    score = 60
                else:
                    score = 40
            
            elif metric_type == "cpu_usage":
                # Menor é melhor (%)
                if avg_value <= 30:
                    score = 100
                elif avg_value <= 50:
                    score = 80
                elif avg_value <= 70:
                    score = 60
                else:
                    score = 40
            
            else:
                score = 75  # Score neutro para tipos desconhecidos
            
            scores.append(score)
        
        # Retornar média dos scores
        return statistics.mean(scores) if scores else 50.0
    
    def _generate_period_insights(self, metrics: List[PerformanceMetric]) -> List[PerformanceInsight]:
        """Gerar insights para um período específico"""
        insights = []
        
        try:
            if not metrics:
                return insights
            
            # Agrupar por tipo
            by_type = {}
            for metric in metrics:
                metric_type = metric.metric_type.value
                if metric_type not in by_type:
                    by_type[metric_type] = []
                by_type[metric_type].append(metric.value)
            
            # Insight: Performance geral
            performance_score = self._calculate_performance_score(metrics)
            if performance_score >= 80:
                insights.append(PerformanceInsight(
                    insight_type="performance_status",
                    title="Performance Excelente",
                    description=f"Sistema operando com performance excelente ({performance_score:.1f}/100)",
                    severity="info",
                    confidence=0.9,
                    impact_score=performance_score,
                    recommendations=["Manter configurações atuais", "Monitorar tendências"],
                    supporting_data={"performance_score": performance_score}
                ))
            elif performance_score < 60:
                insights.append(PerformanceInsight(
                    insight_type="performance_status",
                    title="Performance Abaixo do Esperado",
                    description=f"Performance do sistema está abaixo do ideal ({performance_score:.1f}/100)",
                    severity="warning",
                    confidence=0.8,
                    impact_score=100 - performance_score,
                    recommendations=[
                        "Analisar gargalos de performance",
                        "Considerar otimizações de cache",
                        "Verificar uso de recursos"
                    ],
                    supporting_data={"performance_score": performance_score}
                ))
            
            # Insight: Tempo de execução
            if "execution_time" in by_type:
                exec_times = by_type["execution_time"]
                avg_time = statistics.mean(exec_times)
                
                if avg_time > 120:  # > 2 minutos
                    insights.append(PerformanceInsight(
                        insight_type="execution_time",
                        title="Tempo de Execução Elevado",
                        description=f"Tempo médio de execução está alto ({avg_time:.1f}s)",
                        severity="warning",
                        confidence=0.8,
                        impact_score=min((avg_time / 60) * 10, 100),
                        recommendations=[
                            "Implementar cache mais agressivo",
                            "Otimizar consultas de dados",
                            "Considerar execução paralela"
                        ],
                        supporting_data={"avg_execution_time": avg_time, "max_time": max(exec_times)}
                    ))
            
            # Insight: Cache hit rate
            if "cache_hit_rate" in by_type:
                hit_rates = by_type["cache_hit_rate"]
                avg_hit_rate = statistics.mean(hit_rates)
                
                if avg_hit_rate < 50:  # < 50%
                    insights.append(PerformanceInsight(
                        insight_type="cache_performance",
                        title="Taxa de Cache Hit Baixa",
                        description=f"Cache hit rate está baixo ({avg_hit_rate:.1f}%)",
                        severity="warning",
                        confidence=0.9,
                        impact_score=(50 - avg_hit_rate) * 2,
                        recommendations=[
                            "Revisar estratégia de cache",
                            "Aumentar TTL do cache",
                            "Implementar cache warming"
                        ],
                        supporting_data={"avg_hit_rate": avg_hit_rate}
                    ))
            
            return insights
            
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao gerar insights: {e}")
            return []
    
    def _should_generate_insights(self) -> bool:
        """Verificar se deve gerar insights"""
        elapsed = (datetime.now() - self.last_insight_generation).total_seconds()
        return elapsed >= self.insight_generation_interval
    
    def _should_analyze_trends(self) -> bool:
        """Verificar se deve analisar tendências"""
        elapsed = (datetime.now() - self.last_trend_analysis).total_seconds()
        return elapsed >= 3600  # A cada hora
    
    def _generate_insights(self):
        """Gerar insights automáticos"""
        try:
            with self.lock:
                # Análise das últimas 24 horas
                recent_metrics = self._filter_metrics(
                    datetime.now() - timedelta(hours=24),
                    datetime.now()
                )
                
                # Gerar insights
                new_insights = self._generate_period_insights(recent_metrics)
                
                # Adicionar insights únicos
                for insight in new_insights:
                    if not any(i.insight_type == insight.insight_type and 
                             i.title == insight.title for i in self.insights[-5:]):
                        self.insights.append(insight)
                
                # Manter apenas últimos 50 insights
                if len(self.insights) > 50:
                    self.insights = self.insights[-50:]
                
                self.last_insight_generation = datetime.now()
                
                if new_insights:
                    perf_analytics_logger.info(f"💡 {len(new_insights)} novos insights gerados")
                
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao gerar insights automáticos: {e}")
    
    def _analyze_trends(self):
        """Analisar tendências automáticamente"""
        try:
            with self.lock:
                # Análise de tendências para cada tipo de métrica
                for metric_type in PerformanceMetricType:
                    self._analyze_metric_trend(metric_type)
                
                self.last_trend_analysis = datetime.now()
                
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro na análise de tendências: {e}")
    
    def _analyze_metric_trend(self, metric_type: PerformanceMetricType):
        """Analisar tendência de um tipo específico de métrica"""
        try:
            # Obter métricas das últimas horas configuradas
            window_hours = self.analysis_config["trend_detection_window"]
            start_time = datetime.now() - timedelta(hours=window_hours)
            
            relevant_metrics = []
            for metric_list in self.historical_metrics.values():
                for metric in metric_list:
                    if (metric.metric_type == metric_type and 
                        metric.timestamp >= start_time):
                        relevant_metrics.append(metric)
            
            if len(relevant_metrics) < self.analysis_config["minimum_data_points"]:
                return
            
            # Calcular tendência
            values = [m.value for m in relevant_metrics]
            timestamps = [m.timestamp for m in relevant_metrics]
            
            # Dividir em duas metades para comparar
            mid_point = len(values) // 2
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                
                change_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg != 0 else 0
                
                # Determinar direção
                if abs(change_percent) < self.analysis_config["significant_change_threshold"]:
                    direction = "stable"
                elif change_percent > 0:
                    direction = "improving" if metric_type == PerformanceMetricType.CACHE_HIT_RATE else "degrading"
                else:
                    direction = "degrading" if metric_type == PerformanceMetricType.CACHE_HIT_RATE else "improving"
                
                # Calcular confiança baseada na quantidade de dados
                confidence = min(len(relevant_metrics) / 20.0, 1.0)  # Máximo com 20+ pontos
                
                # Criar tendência
                trend = PerformanceTrend(
                    metric_type=metric_type,
                    timeframe=AnalysisTimeframe.CUSTOM,
                    trend_direction=direction,
                    change_percent=abs(change_percent),
                    current_value=second_avg,
                    previous_value=first_avg,
                    confidence=confidence,
                    data_points=len(relevant_metrics)
                )
                
                trend_key = f"{metric_type.value}_trend"
                self.trends[trend_key] = trend
                
        except Exception as e:
            perf_analytics_logger.warning(f"⚠️ Erro na análise de tendência {metric_type.value}: {e}")
    
    def _cleanup_old_data(self):
        """Limpar dados antigos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            with self.lock:
                # Limpar métricas antigas
                for metric_key in list(self.historical_metrics.keys()):
                    self.historical_metrics[metric_key] = [
                        m for m in self.historical_metrics[metric_key]
                        if m.timestamp > cutoff_date
                    ]
                    
                    # Remove chaves vazias
                    if not self.historical_metrics[metric_key]:
                        del self.historical_metrics[metric_key]
                
                # Limpar buffer
                self.metrics_buffer = [
                    m for m in self.metrics_buffer
                    if m.timestamp > cutoff_date
                ]
                
                # Limpar insights antigos
                insight_cutoff = datetime.now() - timedelta(days=7)  # Manter insights por 7 dias
                # Note: insights não têm timestamp direto, então mantemos os últimos 50
                
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro na limpeza de dados: {e}")
    
    # =============== API PÚBLICA ===============
    
    def get_current_insights(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obter insights atuais"""
        with self.lock:
            insights = self.insights[-20:]  # Últimos 20
            
            if severity:
                insights = [i for i in insights if i.severity == severity]
            
            return [asdict(insight) for insight in insights]
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Obter dados para dashboard de performance"""
        try:
            # Análise das últimas 24 horas
            analysis_24h = self.analyze_performance(AnalysisTimeframe.LAST_24H)
            
            # Tendências atuais
            current_trends = {}
            with self.lock:
                for trend_key, trend in self.trends.items():
                    current_trends[trend_key] = asdict(trend)
            
            # Insights recentes
            recent_insights = self.get_current_insights()
            
            # Benchmarks
            benchmark_data = {}
            with self.lock:
                for bench_name, benchmark in self.benchmarks.items():
                    benchmark_data[bench_name] = asdict(benchmark)
            
            return {
                "last_updated": datetime.now().isoformat(),
                "performance_analysis_24h": analysis_24h,
                "current_trends": current_trends,
                "recent_insights": recent_insights,
                "benchmarks": benchmark_data,
                "system_status": self._get_system_status()
            }
            
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao obter dados do dashboard: {e}")
            return {"error": str(e)}
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Obter status do sistema"""
        with self.lock:
            total_metrics = sum(len(metrics) for metrics in self.historical_metrics.values())
            
            return {
                "total_metrics_collected": total_metrics,
                "active_trends": len(self.trends),
                "total_insights": len(self.insights),
                "benchmarks_count": len(self.benchmarks),
                "last_insight_generation": self.last_insight_generation.isoformat(),
                "real_time_analysis_enabled": self.enable_real_time_analysis
            }
    
    def update_benchmark(self, 
                        benchmark_name: str, 
                        current_value: float,
                        target_value: Optional[float] = None):
        """Atualizar benchmark"""
        try:
            with self.lock:
                if benchmark_name in self.benchmarks:
                    benchmark = self.benchmarks[benchmark_name]
                    old_value = benchmark.current_value
                    
                    benchmark.current_value = current_value
                    benchmark.improvement_percent = ((old_value - current_value) / old_value * 100) if old_value != 0 else 0
                    benchmark.last_updated = datetime.now()
                    
                    if target_value is not None:
                        benchmark.target_value = target_value
                    
                    perf_analytics_logger.info(f"📊 Benchmark atualizado: {benchmark_name} = {current_value}")
                
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao atualizar benchmark: {e}")
    
    def export_analytics_report(self, 
                               timeframe: AnalysisTimeframe = AnalysisTimeframe.LAST_WEEK) -> Dict[str, Any]:
        """Exportar relatório de analytics"""
        try:
            analysis = self.analyze_performance(timeframe)
            
            report = {
                "report_generated": datetime.now().isoformat(),
                "timeframe": timeframe.value,
                "executive_summary": {
                    "performance_score": analysis.get("performance_score", 0),
                    "total_metrics": analysis.get("metrics_count", 0),
                    "key_insights": len(self.get_current_insights("warning")) + len(self.get_current_insights("critical"))
                },
                "detailed_analysis": analysis,
                "trends": {k: asdict(v) for k, v in self.trends.items()},
                "benchmarks": {k: asdict(v) for k, v in self.benchmarks.items()},
                "recommendations": self._generate_report_recommendations(analysis)
            }
            
            return report
            
        except Exception as e:
            perf_analytics_logger.error(f"❌ Erro ao exportar relatório: {e}")
            return {"error": str(e)}
    
    def _generate_report_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Gerar recomendações para relatório"""
        recommendations = []
        
        try:
            performance_score = analysis.get("performance_score", 50)
            
            if performance_score < 60:
                recommendations.append("Implementar otimizações de performance imediatas")
            
            if performance_score < 80:
                recommendations.append("Revisar estratégias de cache e otimização")
            
            # Baseado em tendências
            trends = analysis.get("trends", {})
            for metric_type, trend_data in trends.items():
                if trend_data.get("direction") == "decreasing" and "execution_time" in metric_type:
                    recommendations.append("Investigar causas de degradação no tempo de execução")
                elif trend_data.get("direction") == "decreasing" and "cache_hit_rate" in metric_type:
                    recommendations.append("Otimizar estratégia de cache")
            
            if not recommendations:
                recommendations.append("Manter monitoramento contínuo da performance")
            
            return recommendations
            
        except Exception as e:
            perf_analytics_logger.warning(f"⚠️ Erro ao gerar recomendações: {e}")
            return ["Revisar configurações de performance"] 