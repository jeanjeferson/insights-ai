"""
Dashboard Executivo em Tempo Real para CrewAI Flow
Implementa interface web para monitoramento em tempo real
"""

import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import logging
from dataclasses import asdict

# Configurar logger
dashboard_logger = logging.getLogger('dashboard_realtime')
dashboard_logger.setLevel(logging.INFO)

class RealtimeDashboard:
    """Dashboard executivo em tempo real"""
    
    def __init__(self, monitoring_system=None, recovery_system=None):
        from insights.flow_monitoring import get_global_monitoring_system, MetricType
        from insights.flow_recovery import get_global_recovery_system
        
        self.monitoring_system = monitoring_system or get_global_monitoring_system()
        self.recovery_system = recovery_system or get_global_recovery_system()
        self.MetricType = MetricType  # Tornar MetricType acessível via self
        
        # Configurações do dashboard
        self.refresh_interval = 5  # segundos
        self.max_data_points = 100
        self.active_flows = {}
        
        dashboard_logger.info("📊 RealtimeDashboard inicializado")
    
    def run_dashboard(self, port: int = 8501, host: str = "localhost"):
        """Executar dashboard Streamlit"""
        st.set_page_config(
            page_title="🚀 INSIGHTS-AI | Dashboard Executivo",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self._render_main_dashboard()
    
    def _render_main_dashboard(self):
        """Renderizar dashboard principal"""
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>🚀 INSIGHTS-AI | Dashboard Executivo</h1>
            <p style='color: #666; font-size: 1.1rem;'>
                Monitoramento em Tempo Real • Etapa 3 • Sistema Avançado
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas principais
        self._render_kpi_cards()
        
        # Layout em colunas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_performance_charts()
            self._render_flow_timeline()
        
        with col2:
            self._render_health_status()
            self._render_alerts_panel()
        
        # Seção de detalhes expandidos
        st.markdown("---")
        self._render_detailed_metrics()
        
        # Auto-refresh
        time.sleep(self.refresh_interval)
        st.rerun()
    
    def _render_kpi_cards(self):
        """Renderizar cards de KPIs principais"""
        st.markdown("### 📊 Métricas Principais")
        
        # Obter métricas atuais
        flows_info = self._get_flows_summary()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_flows = len(flows_info)
            st.metric(
                label="🚀 Flows Ativos",
                value=total_flows,
                delta=f"+{total_flows}" if total_flows > 0 else None
            )
        
        with col2:
            active_alerts = len(self.monitoring_system.get_active_alerts())
            alert_delta = "Normal" if active_alerts == 0 else f"⚠️ {active_alerts}"
            st.metric(
                label="🚨 Alertas Ativos",
                value=active_alerts,
                delta=alert_delta,
                delta_color="inverse" if active_alerts > 0 else "normal"
            )
        
        with col3:
            health_status = self.monitoring_system.get_health_status()
            overall_health = health_status.get("overall_status", "unknown")
            health_emoji = {"healthy": "✅", "warning": "⚠️", "error": "❌"}.get(overall_health, "❓")
            st.metric(
                label="🏥 Status Geral",
                value=f"{health_emoji} {overall_health.title()}",
                delta="Operacional" if overall_health == "healthy" else "Verificar"
            )
        
        with col4:
            # CPU médio dos flows ativos
            cpu_metrics = self.monitoring_system.get_current_metrics(
                metric_type=self.MetricType.RESOURCE,
                minutes_back=5
            )
            cpu_values = [m.value for m in cpu_metrics if m.metric_name == "cpu_usage"]
            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            
            st.metric(
                label="💻 CPU Médio",
                value=f"{avg_cpu:.1f}%",
                delta="Normal" if avg_cpu < 70 else "Alto"
            )
        
        with col5:
            # Throughput de métricas
            total_metrics = len(self.monitoring_system.metrics_buffer)
            st.metric(
                label="📈 Métricas/Min",
                value=f"{total_metrics//60 if total_metrics > 60 else total_metrics}",
                delta=f"Buffer: {total_metrics}"
            )
    
    def _render_performance_charts(self):
        """Renderizar gráficos de performance"""
        st.markdown("### 📈 Performance em Tempo Real")
        
        # Obter métricas dos últimos 30 minutos
        metrics = self.monitoring_system.get_current_metrics(minutes_back=30)
        
        if not metrics:
            st.info("📊 Aguardando dados de métricas...")
            return
        
        # Converter para DataFrame
        df_metrics = pd.DataFrame([
            {
                "timestamp": datetime.fromisoformat(m.timestamp),
                "metric_name": m.metric_name,
                "value": m.value,
                "flow_id": m.flow_id,
                "unit": m.unit
            }
            for m in metrics
        ])
        
        # Gráfico de CPU e Memória
        fig_resources = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
            vertical_spacing=0.1
        )
        
        # CPU
        cpu_data = df_metrics[df_metrics["metric_name"] == "cpu_usage"]
        if not cpu_data.empty:
            fig_resources.add_trace(
                go.Scatter(
                    x=cpu_data["timestamp"],
                    y=cpu_data["value"],
                    mode='lines+markers',
                    name='CPU %',
                    line=dict(color='#FF6B6B')
                ),
                row=1, col=1
            )
        
        # Memória
        memory_data = df_metrics[df_metrics["metric_name"] == "memory_usage"]
        if not memory_data.empty:
            fig_resources.add_trace(
                go.Scatter(
                    x=memory_data["timestamp"],
                    y=memory_data["value"],
                    mode='lines+markers',
                    name='Memory %',
                    line=dict(color='#4ECDC4')
                ),
                row=2, col=1
            )
        
        fig_resources.update_layout(
            height=400,
            showlegend=False,
            title_text="Recursos do Sistema"
        )
        
        st.plotly_chart(fig_resources, use_container_width=True)
        
        # Gráfico de tempo de execução das análises
        exec_time_data = df_metrics[df_metrics["metric_name"].str.contains("execution_time", na=False)]
        
        if not exec_time_data.empty:
            fig_exec = px.bar(
                exec_time_data.groupby("metric_name")["value"].mean().reset_index(),
                x="metric_name",
                y="value",
                title="Tempo Médio de Execução por Análise",
                labels={"value": "Tempo (segundos)", "metric_name": "Análise"}
            )
            fig_exec.update_layout(height=300)
            st.plotly_chart(fig_exec, use_container_width=True)
    
    def _render_flow_timeline(self):
        """Renderizar timeline dos flows"""
        st.markdown("### ⏱️ Timeline de Execução")
        
        flows_summary = self._get_flows_summary()
        
        if not flows_summary:
            st.info("📅 Nenhum flow em execução no momento")
            return
        
        # Criar timeline
        timeline_data = []
        
        for flow_id, info in flows_summary.items():
            start_time = info.get("start_time", time.time())
            current_time = time.time()
            duration = current_time - start_time
            
            timeline_data.append({
                "Flow ID": flow_id,
                "Duração (min)": duration / 60,
                "Status": "🟢 Ativo",
                "Início": datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
            })
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            # Gráfico de barras horizontais
            fig_timeline = px.bar(
                df_timeline,
                x="Duração (min)",
                y="Flow ID",
                orientation='h',
                title="Duração de Execução dos Flows",
                color="Duração (min)",
                color_continuous_scale="Viridis"
            )
            fig_timeline.update_layout(height=max(300, len(timeline_data) * 50))
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Tabela de detalhes
            st.dataframe(df_timeline, use_container_width=True)
    
    def _render_health_status(self):
        """Renderizar status de saúde"""
        st.markdown("### 🏥 Status de Saúde")
        
        health_status = self.monitoring_system.get_health_status()
        
        # Status geral
        overall_status = health_status.get("overall_status", "unknown")
        status_colors = {
            "healthy": "🟢",
            "warning": "🟡", 
            "error": "🔴",
            "unknown": "⚪"
        }
        
        st.markdown(f"""
        **Status Geral:** {status_colors.get(overall_status, '⚪')} {overall_status.title()}
        """)
        
        # Componentes
        components = health_status.get("components", {})
        
        if components:
            st.markdown("**Componentes:**")
            
            for component_key, component_info in components.items():
                status = component_info.get("status", "unknown")
                response_time = component_info.get("response_time", 0)
                
                status_icon = status_colors.get(status, "⚪")
                
                st.markdown(f"""
                - {status_icon} **{component_key}**: {status} 
                  ⏱️ {response_time:.3f}s
                """)
        
        # Recovery Status
        st.markdown("### 🛡️ Sistema de Recovery")
        
        # Buscar status de recovery para flows ativos
        recovery_info = {"checkpoints": 0, "failures": 0}
        
        for flow_id in self.monitoring_system.current_flows.keys():
            recovery_status = self.recovery_system.get_recovery_status(flow_id)
            recovery_info["checkpoints"] += recovery_status.get("total_checkpoints", 0)
            recovery_info["failures"] += recovery_status.get("total_failures", 0)
        
        st.markdown(f"""
        - 📸 **Checkpoints**: {recovery_info['checkpoints']}
        - ❌ **Falhas**: {recovery_info['failures']}
        - 🔄 **Auto-Recovery**: {'✅ Ativo' if self.recovery_system.auto_recovery_enabled else '❌ Inativo'}
        """)
    
    def _render_alerts_panel(self):
        """Renderizar painel de alertas"""
        st.markdown("### 🚨 Alertas Ativos")
        
        active_alerts = self.monitoring_system.get_active_alerts()
        
        if not active_alerts:
            st.success("✅ Nenhum alerta ativo")
            return
        
        # Separar por nível
        alert_levels = {"critical": [], "error": [], "warning": [], "info": []}
        
        for alert in active_alerts:
            level = alert.level.value
            if level in alert_levels:
                alert_levels[level].append(alert)
        
        # Mostrar alertas por prioridade
        level_colors = {
            "critical": "🚨",
            "error": "❌", 
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        for level, alerts in alert_levels.items():
            if alerts:
                st.markdown(f"**{level_colors[level]} {level.title()} ({len(alerts)})**")
                
                for alert in alerts[:5]:  # Mostrar apenas os 5 mais recentes
                    timestamp = datetime.fromisoformat(alert.timestamp)
                    time_ago = datetime.now() - timestamp
                    
                    with st.expander(f"{alert.title} ({time_ago.seconds//60}min atrás)"):
                        st.markdown(f"""
                        **Mensagem:** {alert.message}
                        
                        **Flow ID:** {alert.flow_id}
                        
                        **Métrica:** {alert.metric_name}
                        
                        **Valor Atual:** {alert.current_value}
                        
                        **Threshold:** {alert.threshold_value}
                        
                        **Timestamp:** {timestamp.strftime("%d/%m/%Y %H:%M:%S")}
                        """)
                        
                        # Botão para resolver alerta
                        if st.button(f"Resolver", key=f"resolve_{alert.alert_id}"):
                            self.monitoring_system.resolve_alert(alert.alert_id)
                            st.success("Alerta resolvido!")
                            st.rerun()
    
    def _render_detailed_metrics(self):
        """Renderizar métricas detalhadas"""
        st.markdown("### 📊 Métricas Detalhadas")
        
        # Tabs para diferentes categorias
        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Performance", "💻 Recursos", "🏥 Saúde", "📈 Business"])
        
        with tab1:
            self._render_performance_metrics()
        
        with tab2:
            self._render_resource_metrics()
        
        with tab3:
            self._render_health_metrics()
        
        with tab4:
            self._render_business_metrics()
    
    def _render_performance_metrics(self):
        """Renderizar métricas de performance"""
        st.markdown("#### 🚀 Métricas de Performance")
        
        # Obter métricas de performance
        perf_metrics = self.monitoring_system.get_current_metrics(
                            metric_type=self.MetricType.PERFORMANCE,
            minutes_back=60
        )
        
        if not perf_metrics:
            st.info("📊 Nenhuma métrica de performance disponível")
            return
        
        # Criar DataFrame
        df_perf = pd.DataFrame([
            {
                "Timestamp": datetime.fromisoformat(m.timestamp),
                "Métrica": m.metric_name,
                "Valor": m.value,
                "Unidade": m.unit,
                "Flow ID": m.flow_id
            }
            for m in perf_metrics
        ])
        
        # Mostrar tabela
        st.dataframe(df_perf.sort_values("Timestamp", ascending=False), use_container_width=True)
        
        # Estatísticas
        if not df_perf.empty:
            st.markdown("**Estatísticas:**")
            stats = df_perf.groupby("Métrica")["Valor"].agg(['count', 'mean', 'min', 'max', 'std'])
            st.dataframe(stats, use_container_width=True)
    
    def _render_resource_metrics(self):
        """Renderizar métricas de recursos"""
        st.markdown("#### 💻 Métricas de Recursos")
        
        # Métricas atuais de recursos
        resource_metrics = self.monitoring_system.get_current_metrics(
                            metric_type=self.MetricType.RESOURCE,
            minutes_back=30
        )
        
        if not resource_metrics:
            st.info("📊 Nenhuma métrica de recursos disponível")
            return
        
        # Últimas métricas por tipo
        latest_metrics = {}
        for metric in resource_metrics:
            latest_metrics[metric.metric_name] = metric.value
        
        # Cards de recursos
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cpu_usage = latest_metrics.get("cpu_usage", 0)
            st.metric(
                "CPU Usage",
                f"{cpu_usage:.1f}%",
                delta="Normal" if cpu_usage < 70 else "Alto",
                delta_color="normal" if cpu_usage < 70 else "inverse"
            )
        
        with col2:
            memory_usage = latest_metrics.get("memory_usage", 0)
            st.metric(
                "Memory Usage", 
                f"{memory_usage:.1f}%",
                delta="Normal" if memory_usage < 80 else "Alto",
                delta_color="normal" if memory_usage < 80 else "inverse"
            )
        
        with col3:
            disk_usage = latest_metrics.get("disk_usage", 0)
            st.metric(
                "Disk Usage",
                f"{disk_usage:.1f}%", 
                delta="Normal" if disk_usage < 85 else "Alto",
                delta_color="normal" if disk_usage < 85 else "inverse"
            )
        
        # Gráfico histórico
        if resource_metrics:
            df_resources = pd.DataFrame([
                {
                    "Timestamp": datetime.fromisoformat(m.timestamp),
                    "Métrica": m.metric_name,
                    "Valor": m.value
                }
                for m in resource_metrics
                if m.metric_name in ["cpu_usage", "memory_usage", "disk_usage"]
            ])
            
            if not df_resources.empty:
                fig = px.line(
                    df_resources,
                    x="Timestamp",
                    y="Valor",
                    color="Métrica",
                    title="Histórico de Recursos (30min)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_health_metrics(self):
        """Renderizar métricas de saúde"""
        st.markdown("#### 🏥 Métricas de Saúde")
        
        # Health checks recentes
        health_checks = list(self.monitoring_system.health_checks.values())
        
        if not health_checks:
            st.info("🏥 Nenhum health check disponível")
            return
        
        # Criar DataFrame dos health checks
        df_health = pd.DataFrame([
            {
                "Timestamp": datetime.fromisoformat(hc.timestamp),
                "Flow ID": hc.flow_id,
                "Componente": hc.component,
                "Status": hc.status,
                "Response Time": f"{hc.response_time:.3f}s",
                "Detalhes": len(hc.details.get("checks", []))
            }
            for hc in health_checks
        ])
        
        st.dataframe(df_health.sort_values("Timestamp", ascending=False), use_container_width=True)
        
        # Distribuição de status
        if not df_health.empty:
            status_counts = df_health["Status"].value_counts()
            
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Distribuição de Status dos Health Checks"
            )
            st.plotly_chart(fig_status, use_container_width=True)
    
    def _render_business_metrics(self):
        """Renderizar métricas de negócio"""
        st.markdown("#### 📈 Métricas de Negócio")
        
        # Métricas de negócio
        business_metrics = self.monitoring_system.get_current_metrics(
                            metric_type=self.MetricType.BUSINESS,
            minutes_back=120
        )
        
        if not business_metrics:
            st.info("📈 Nenhuma métrica de negócio disponível")
            return
        
        # Análises por tipo
        analysis_metrics = {}
        success_rates = {}
        
        for metric in business_metrics:
            if "success_rate" in metric.metric_name:
                analysis_name = metric.metric_name.replace("_success_rate", "")
                success_rates[analysis_name] = metric.value
            else:
                if metric.metric_name not in analysis_metrics:
                    analysis_metrics[metric.metric_name] = []
                analysis_metrics[metric.metric_name].append(metric.value)
        
        # Cards de taxa de sucesso
        if success_rates:
            st.markdown("**Taxa de Sucesso das Análises:**")
            
            cols = st.columns(min(len(success_rates), 4))
            
            for i, (analysis, rate) in enumerate(success_rates.items()):
                with cols[i % 4]:
                    st.metric(
                        analysis.replace("_", " ").title(),
                        f"{rate*100:.1f}%",
                        delta="Excelente" if rate > 0.95 else "Bom" if rate > 0.8 else "Atenção"
                    )
        
        # Outras métricas de negócio
        if analysis_metrics:
            st.markdown("**Outras Métricas:**")
            
            for metric_name, values in analysis_metrics.items():
                if values:
                    avg_value = sum(values) / len(values)
                    st.metric(
                        metric_name.replace("_", " ").title(),
                        f"{avg_value:.2f}",
                        delta=f"{len(values)} medições"
                    )
    
    def _get_flows_summary(self) -> Dict[str, Any]:
        """Obter resumo dos flows ativos"""
        return self.monitoring_system.current_flows.copy()
    
    def export_dashboard_data(self) -> Dict[str, Any]:
        """Exportar dados do dashboard"""
        return {
            "timestamp": datetime.now().isoformat(),
            "flows_summary": self._get_flows_summary(),
            "health_status": self.monitoring_system.get_health_status(),
            "active_alerts": [alert.to_dict() for alert in self.monitoring_system.get_active_alerts()],
            "recent_metrics": [
                metric.to_dict() for metric in 
                self.monitoring_system.get_current_metrics(minutes_back=60)
            ]
        }


# =============== FUNÇÕES DE UTILIDADE ===============

def launch_dashboard(port: int = 8501, monitoring_system=None, recovery_system=None):
    """Lançar dashboard em thread separada"""
    dashboard = RealtimeDashboard(monitoring_system, recovery_system)
    
    # Configurar e executar Streamlit
    import subprocess
    import sys
    
    dashboard_logger.info(f"🚀 Lançando dashboard na porta {port}")
    
    # Criar arquivo temporário do dashboard
    dashboard_file = Path("temp_dashboard.py")
    dashboard_content = f"""
import streamlit as st
from insights.dashboard_realtime import RealtimeDashboard

if __name__ == "__main__":
    dashboard = RealtimeDashboard()
    dashboard.run_dashboard()
"""
    
    dashboard_file.write_text(dashboard_content)
    
    try:
        # Executar Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_file), 
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    finally:
        # Limpar arquivo temporário
        if dashboard_file.exists():
            dashboard_file.unlink()

def create_dashboard_app():
    """Criar app Streamlit standalone"""
    dashboard = RealtimeDashboard()
    dashboard.run_dashboard()

# Para execução direta
if __name__ == "__main__":
    create_dashboard_app() 